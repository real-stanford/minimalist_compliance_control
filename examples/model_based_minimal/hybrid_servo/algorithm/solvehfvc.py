"""
Hybrid Force-Velocity Control (HFVC) solver.

Python implementation following the MATLAB algorithm structure from
https://github.com/yifan-hou/hybrid_servoing/blob/master/algorithm/matlab/solvehfvc.m
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg

try:
    from cvxopt import matrix, solvers
except Exception:
    matrix = None  # type: ignore[assignment]
    solvers = None  # type: ignore[assignment]

np.set_printoptions(precision=4, suppress=True)


@dataclass
class HFVC:
    """
    Hybrid Force-Velocity Control action structure.

    Attributes:
        n_av: Number of velocity controlled actions
        n_af: Number of force controlled actions
        R_a: Transformation matrix for control actions
        w_av: Velocity control magnitudes
        eta_af: Force control magnitudes
    """

    n_av: int
    n_af: int
    R_a: np.ndarray
    w_av: np.ndarray
    eta_af: np.ndarray


def solvehfvc(
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
) -> HFVC:
    """
    Solve hybrid force-velocity control problem.

    This is a complete Python reimplementation following the exact MATLAB algorithm structure.

    Args:
        N_all: Contact normal matrix (kDimContactForce × kDimGeneralized)
        G: Holonomic constraint matrix
        b_G: Holonomic constraint vector
        F: External force vector
        Aeq: Equality constraint matrix (guard conditions)
        beq: Equality constraint vector
        A: Inequality constraint matrix (friction cones, etc.)
        b_A: Inequality constraint vector
        kDimActualized: Number of actuated DOF
        kDimUnActualized: Number of unactuated DOF
        kDimSlidingFriction: Number of sliding friction constraints
        kDimLambda: Number of contact force variables
        kNumSeeds: Number of random seeds for optimization
        kPrintLevel: Print level (0=quiet, 1=basic, 2=verbose)

    Returns:
        HFVC: Control action structure with n_av, n_af, R_a, w_av, eta_af
    """
    if matrix is None or solvers is None:
        raise ImportError(
            "cvxopt is required for solvehfvc(). Install with: pip install cvxopt"
        )

    # ==================== Problem Setup ====================
    kDimGeneralized = kDimActualized + kDimUnActualized
    kDimContactForce = kDimLambda + kDimSlidingFriction

    if kPrintLevel >= 2:
        print("Begin solving for velocity commands")
        print("  [1] Determine Possible Dimension of control")

    # Input validation
    assert N_all.shape[1] == kDimGeneralized
    assert N_all.shape[0] == kDimContactForce
    assert G.shape[0] == b_G.shape[0]
    assert G.shape[1] == kDimGeneralized
    assert F.shape[0] == kDimGeneralized

    # Extract contact constraint matrix N (excluding sliding friction)
    N = N_all[:kDimLambda, :]
    NG = np.vstack([N, G])

    # ==================== Matrix Decomposition ====================
    rank_N = np.linalg.matrix_rank(N)
    rank_NG = np.linalg.matrix_rank(NG)
    basis_N = scipy.linalg.null_space(N)

    assert rank_NG > 0

    # Determine control dimensions
    n_av = rank_NG - rank_N
    n_af = kDimActualized - n_av

    # Feasibility check
    assert rank_N + kDimActualized >= kDimGeneralized

    # ==================== Control Direction Computation ====================
    if n_av == 0:
        if kPrintLevel >= 2:
            print("  [2] No feasible velocity control can satisfy the goal")
        R_a = np.eye(kDimActualized)
        T = np.eye(kDimGeneralized)
        w_av = np.array([])
    else:
        if kPrintLevel >= 2:
            print("  [2] Solving for Directions by PGD")
        null_NG = scipy.linalg.null_space(NG)
        C_c = np.vstack(
            [
                null_NG.T,
                np.hstack(
                    [
                        np.eye(kDimUnActualized),
                        np.zeros((kDimUnActualized, kDimActualized)),
                    ]
                ),
            ]
        )
        basis_c = scipy.linalg.null_space(C_c)
        b_NG = np.concatenate([np.zeros(N.shape[0]), b_G.ravel()])
        v_star = np.linalg.lstsq(NG, b_NG, rcond=None)[0]
        # ==================== Projected Gradient Descent ====================
        NIter = 50
        n_c = rank_NG - kDimUnActualized
        BB = basis_c.T @ basis_c
        NN = basis_N @ basis_N.T
        k_all = []
        cost_all = np.zeros(kNumSeeds)
        k_all = np.random.rand(n_c, n_av, kNumSeeds)
        # Multiple random initializations
        for seed in range(kNumSeeds):
            k = k_all[:, :, seed]
            bck = basis_c @ k
            for i in range(bck.shape[1]):
                bck_col_norm = np.linalg.norm(bck[:, i])
                if bck_col_norm > 1e-12:
                    k[:, i] /= bck_col_norm
            for iteration in range(NIter):
                g = np.zeros((n_c, n_av))
                costs = 0.0

                for i in range(n_av):
                    for j in range(n_av):
                        if i == j:
                            continue
                        tempcost = np.linalg.norm(k[:, i].T @ BB @ k[:, j])
                        costs += tempcost * tempcost
                        g[:, i] += 2.0 * (k[:, i].T @ BB @ k[:, j]) * BB @ k[:, j]

                    g[:, i] -= 2.0 * basis_c.T @ NN @ basis_c @ k[:, i]
                    costs -= k[:, i].T @ basis_c.T @ NN @ basis_c @ k[:, i]

                k -= 0.1 * g
                bck = basis_c @ k
                for i in range(bck.shape[1]):
                    bck_col_norm = np.linalg.norm(bck[:, i])
                    if bck_col_norm > 1e-12:
                        k[:, i] /= bck_col_norm

            cost_all[seed] = costs

        min_id = np.argmin(cost_all)
        k_best = k_all[:, :, min_id]
        C_best = (basis_c @ k_best).T
        C_best_actualized = C_best[:, -kDimActualized:]
        basis_C_best_actualized = scipy.linalg.null_space(C_best_actualized)
        rank_C_best_actualized = np.linalg.matrix_rank(C_best_actualized)
        assert rank_C_best_actualized > 0

        R_a = np.vstack([basis_C_best_actualized.T, C_best_actualized])
        T = np.zeros((kDimGeneralized, kDimGeneralized))
        T[:kDimUnActualized, :kDimUnActualized] = np.eye(kDimUnActualized)
        T[-kDimActualized:, -kDimActualized:] = R_a
        w_av = C_best @ v_star

    # ==================== Force Command Computation ====================
    if kPrintLevel >= 2:
        print("Begin Solving for force commands.")

    # Unactuated dimensions constraint matrix
    H = np.zeros((kDimUnActualized, kDimGeneralized))
    H[:, :kDimUnActualized] = np.eye(kDimUnActualized)
    T_inv = np.linalg.inv(T)

    M_newton_H = np.hstack([np.zeros((kDimUnActualized, kDimContactForce)), H @ T_inv])
    M_newton_N = np.hstack([T @ N_all.T, np.eye(kDimGeneralized)])
    M_newton = np.vstack([M_newton_H, M_newton_N, Aeq])
    b_newton = np.concatenate([np.zeros(H.shape[0]), (-T @ F).ravel(), beq])
    M_free = np.hstack(
        [
            M_newton[:, : kDimContactForce + kDimUnActualized],
            M_newton[:, kDimContactForce + kDimUnActualized + n_af :],
        ]
    )

    M_eta_af = M_newton[
        :,
        kDimContactForce + kDimUnActualized : kDimContactForce
        + kDimUnActualized
        + n_af,
    ]

    # ==================== Quadratic Programming Setup ====================
    n_free = kDimContactForce + kDimUnActualized + n_av
    n_dual_free = M_newton.shape[0]

    # Cost matrix - regularized to minimize force magnitudes
    Gdiag = np.zeros(n_free + n_dual_free + n_af)
    Gdiag[-n_af:] = 1.0  # Penalty on force variables
    Q = np.diag(Gdiag)
    f = np.zeros(n_free + n_dual_free + n_af)
    qpAeq_top = np.hstack([2 * np.eye(n_free), M_free.T, np.zeros((n_free, n_af))])
    qpAeq_bottom = np.hstack(
        [M_free, np.zeros((M_free.shape[0], M_free.shape[0])), M_eta_af]
    )
    qpAeq = np.vstack([qpAeq_top, qpAeq_bottom])

    A_temp = np.hstack([A[:, :kDimContactForce], A[:, kDimContactForce:] @ T_inv])
    A_lambda_eta_u = A_temp[:, : kDimContactForce + kDimUnActualized]
    A_eta_af = A_temp[
        :,
        kDimContactForce + kDimUnActualized : kDimContactForce
        + kDimUnActualized
        + n_af,
    ]
    qpA_parts = [A_lambda_eta_u]
    A_eta_av = A_temp[:, kDimContactForce + kDimUnActualized + n_af :]
    qpA_parts.append(A_eta_av)
    qpA_parts.extend([np.zeros((A.shape[0], n_dual_free)), A_eta_af])
    qpA = np.hstack(qpA_parts)

    qpbeq = np.concatenate([np.zeros(n_free), b_newton])

    # Inequality constraints
    qpb = b_A

    # ==================== Solve QP using scipy.optimize.minimize ====================
    solvers.options["show_progress"] = False
    solvers.options["refinement"] = 2
    # print("rank Aeq:", np.linalg.matrix_rank(qpAeq))
    # print("rank A:", np.linalg.matrix_rank(qpA))
    # print("rank Q:", np.linalg.matrix_rank(Q))
    # print("A shape:", qpA.shape)
    # print("Aeq shape:", qpAeq.shape)
    # print("Q shape:", Q)
    # PAG = np.vstack([Q, qpAeq, qpA])
    # print("PAG rank:", np.linalg.matrix_rank(PAG))
    # print("PAG shape:", PAG.shape)
    # print("n:", Q.shape[0])
    # print("p:", Aeq.shape[0])
    solution = solvers.qp(
        matrix(Q), matrix(f), matrix(qpA), matrix(qpb), matrix(qpAeq), matrix(qpbeq)
    )
    x = solution["x"]
    # status = solution["status"]
    # iterations = solution["iterations"]
    eta_af = x[n_free + n_dual_free :]
    eta_af = np.array(eta_af)
    return HFVC(n_av=n_av, n_af=n_af, R_a=R_a, w_av=w_av, eta_af=eta_af)


def transform_hfvc_to_global(hfvc_solution: HFVC):
    n_av = hfvc_solution.n_av
    n_af = hfvc_solution.n_af
    R_a = hfvc_solution.R_a
    w_av = hfvc_solution.w_av
    eta_af = hfvc_solution.eta_af
    vec1 = np.concatenate([np.zeros((n_af, 1)), w_av.reshape(-1, 1)], axis=0)
    global_velocity = np.linalg.inv(R_a) @ vec1
    vec2 = np.concatenate([eta_af.reshape(-1, 1), np.zeros((n_av, 1))], axis=0)
    global_force = np.linalg.inv(R_a) @ vec2
    return global_velocity, global_force
