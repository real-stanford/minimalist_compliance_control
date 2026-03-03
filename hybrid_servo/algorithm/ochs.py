"""
OCHS (Optimal Contact Hybrid Servoing) solver.

Python implementation following the MATLAB algorithm structure from
ochs.m. This algorithm optimally determines the dimensionality of velocity
control actions based on feasibility constraints.
"""

import numpy as np
import scipy.linalg
import scipy.sparse
from qpsolvers import solve_qp

from hybrid_servo.algorithm.solvehfvc import HFVC

# from toddlerbot.utils.misc_utils import profile

np.set_printoptions(precision=4, suppress=True)


# @profile()
def solve_ochs(
    N_all: np.ndarray,
    G: np.ndarray,
    b_G: np.ndarray,
    F: np.ndarray,
    Aeq: np.ndarray,
    beq: np.ndarray,
    A: np.ndarray,
    b_A: np.ndarray,
    kDimActualized: int,
    kDimUnActualized: int,
    kDimSlidingFriction: int,
    kDimLambda: int,
    kNumSeeds: int = 3,
    kPrintLevel: int = 0,
    J_All=None,
) -> HFVC:
    """
    Solve OCHS problem with optimal velocity control dimensions.

    This algorithm optimally determines the number of velocity-controlled DOF
    based on feasibility analysis, choosing between maximum available dimensions
    and minimum required dimensions.

    Args:
        N_all: Contact normal matrix (kDimContactForce × kDimGeneralized)
        G: Holonomic constraint matrix
        b_G: Holonomic constraint vector
        F: External force vector
        Aeq: Equality constraint matrix (guard conditions) - not used in OCHS
        beq: Equality constraint vector - not used in OCHS
        A: Inequality constraint matrix (friction cones, etc.)
        b_A: Inequality constraint vector
        kDimActualized: Number of actuated DOF
        kDimUnActualized: Number of unactuated DOF
        kDimSlidingFriction: Number of sliding friction constraints
        kDimLambda: Number of contact force variables
        kNumSeeds: Number of random seeds (not used in this algorithm)
        kPrintLevel: Print level (0=quiet, 1=basic, 2=verbose)

    Returns:
        HFVC: Control action structure with n_av, n_af, R_a, w_av, eta_af
    """

    # ==================== Problem Setup ====================
    kDimGeneralized = kDimActualized + kDimUnActualized
    kDimContactForce = kDimLambda + kDimSlidingFriction

    if kPrintLevel >= 2:
        print("Begin solving OCHS for velocity commands")
        print("  [1] Determine Optimal Dimension of velocity control")

    # Input validation
    assert N_all.shape[1] == kDimGeneralized
    assert N_all.shape[0] == kDimContactForce
    assert G.shape[0] == b_G.shape[0]
    assert G.shape[1] == kDimGeneralized
    assert F.shape[0] == kDimGeneralized

    # Extract contact constraint matrix J (excluding sliding friction)
    J = N_all[:kDimLambda, :]

    # ==================== Constants ====================
    n_a = kDimActualized
    n_u = kDimUnActualized
    n = n_a + n_u

    # Selection matrix M: [zeros(n_u, n_a); eye(n_a)]
    M = np.vstack([np.zeros((n_u, n_a)), np.eye(n_a)])

    # Null space of contact constraints
    U = scipy.linalg.null_space(J).T
    rank_J = n - U.shape[0]

    if U.size == 0:
        # No null space, environment fully constrains the problem
        if kPrintLevel >= 1:
            print("No null space - environment fully constrained")
        return HFVC(
            n_av=0, n_af=n_a, R_a=np.eye(n_a), w_av=np.array([]), eta_af=np.zeros(n_a)
        )

    # Project onto actuated space
    U_bar = U @ M

    # ==================== Null Space HFVC Analysis ====================
    # Get orthogonal basis for actuated null space
    U_bar_n = scipy.linalg.orth(U_bar.T).T

    n_av_max = U_bar_n.shape[0]

    # Analyze feasibility with goal constraints
    JG = np.vstack([J, G])
    b_JG = np.concatenate([np.zeros(J.shape[0]), b_G.ravel()])
    null_JG = scipy.linalg.null_space(JG)
    rank_JG = n - null_JG.shape[1]

    if rank_JG - rank_J > n_av_max:
        # Infeasible problem: goal cannot be satisfied
        if kPrintLevel >= 1:
            print("Infeasible: goal cannot be satisfied")
        return HFVC(
            n_av=0, n_af=n_a, R_a=np.eye(n_a), w_av=np.array([]), eta_af=np.zeros(n_a)
        )

    # ==================== Optimal Dimension Selection ====================
    if rank_JG - rank_J == n_av_max:
        # Must use all free DOF for velocity control
        if kPrintLevel >= 2:
            print("  Using maximum velocity control dimensions")
        n_av = n_av_max
        C_bar = U_bar_n
    else:
        # Don't need to use up all free space for velocity control
        if kPrintLevel >= 2:
            print("  Using minimal required velocity control dimensions")
        n_av = rank_JG - rank_J

        # Compute optimal control directions using null space projection
        # K = ([zeros(n_av_max, n_u) U_bar_n] * null_JG)'
        temp_matrix = np.hstack([np.zeros((n_av_max, n_u)), U_bar_n])
        K = (temp_matrix @ null_JG).T

        # Find null space of K
        k = scipy.linalg.null_space(K)

        if k.shape[1] < n_av:
            # No solution - cannot achieve the goal
            if kPrintLevel >= 1:
                print("No solution - cannot achieve goal with minimal dimensions")
            return HFVC(
                n_av=0,
                n_af=n_a,
                R_a=np.eye(n_a),
                w_av=np.array([]),
                eta_af=np.zeros(n_a),
            )

        # Use first n_av columns to construct C_bar
        C_bar = k[:, :n_av].T @ U_bar_n

    # ==================== Control Direction Setup ====================
    C = np.hstack([np.zeros((n_av, n_u)), C_bar])
    n_af = n_a - n_av

    # Construct transformation matrices
    R_a = np.vstack([scipy.linalg.null_space(C_bar).T, C_bar])
    T = scipy.linalg.block_diag(np.eye(n_u), R_a)

    # ==================== Solve for velocity commands ====================
    JC = np.vstack([J, C])
    JCG = np.vstack([JC, G])

    if np.linalg.matrix_rank(JC) < np.linalg.matrix_rank(JCG):
        # Infeasible goal
        if kPrintLevel >= 1:
            print("Infeasible goal - rank condition failed")
        return HFVC(
            n_av=0, n_af=n_a, R_a=np.eye(n_a), w_av=np.array([]), eta_af=np.zeros(n_a)
        )

    # Solve for reference velocity
    v_star, residuals, _, _ = np.linalg.lstsq(JG, b_JG, rcond=None)
    res_norm = np.linalg.norm(JG @ v_star - b_JG)
    if kPrintLevel >= 1:
        print(f"lstsq residual norm: {res_norm}")
    w_av = C @ v_star

    if kPrintLevel >= 2:
        print(f"  Velocity control dimensions: n_av = {n_av}, n_af = {n_af}")
        print(f"  Velocity commands: w_av = {w_av}")

    # ==================== Force Command Computation ====================
    if kPrintLevel >= 2:
        print("Begin solving OCHS for force commands")

    if n_af == 0:
        # Pure velocity control
        eta_af = np.array([])
        return HFVC(n_av=n_av, n_af=n_af, R_a=R_a, w_av=w_av, eta_af=eta_af)

    # Unactuated dimensions constraint matrix
    H = np.zeros((n_u, kDimGeneralized))
    H[:, :n_u] = np.eye(n_u)

    # Setup QP for force optimization
    nLambda = kDimContactForce

    # Equality constraints: Newton's laws
    # Aeq_lp = [T*J_All' [zeros(n_u, n_a); eye(n_a)]]
    Aeq_lp = np.hstack([T @ N_all.T, np.vstack([np.zeros((n_u, n_a)), np.eye(n_a)])])
    beq_lp = -T @ F

    # Inequality constraints: friction cones
    A_lp = np.hstack([A, np.zeros((A.shape[0], n_a))])
    b_lp = b_A

    # QP cost: minimize force magnitudes
    Q = scipy.sparse.eye(nLambda + n_a, format="csc")
    f = np.zeros(nLambda + n_a)

    # ==================== Solve QP (prefer qpsolvers+OSQP) ====================
    A_lp_sparse = scipy.sparse.csc_matrix(A_lp)
    Aeq_lp_sparse = scipy.sparse.csc_matrix(Aeq_lp)
    x = solve_qp(
        Q,
        f,
        G=A_lp_sparse,
        h=b_lp,
        A=Aeq_lp_sparse,
        b=beq_lp,
        solver="osqp",
    )
    status = "optimal" if x is not None else "unknown"

    if status == "optimal":
        eta_af = np.array(x[nLambda : nLambda + n_af]).flatten()
    else:
        if kPrintLevel >= 1:
            print(f"QP solver failed with status: {status}")
        eta_af = np.zeros(n_af)

    if kPrintLevel >= 2:
        print(f"  Force commands: eta_af = {eta_af}")
    return HFVC(n_av=n_av, n_af=n_af, R_a=R_a, w_av=w_av, eta_af=eta_af)
