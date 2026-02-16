"""OCHS helper functions used by LEAP rotate compliance policy.

Copied from toddlerbot_internal's
`hybrid_servo/demo/multi_finger_rotate_anything/multi_finger_rotate_anything.py`
with only the functions required by leap rotate policy.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import sympy as sp
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return sp.Matrix(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_inv(q):
    w, x, y, z = q
    return sp.Matrix([w, -x, -y, -z])


def quat_on_vec(v, q):
    v_as_quat = sp.Matrix([0, v[0], v[1], v[2]])
    vec = quat_mul(quat_mul(q, v_as_quat), quat_inv(q))[1:]
    return sp.Matrix(vec)


def generate_constraint_jacobian():
    p_WO = sp.symbols("p_WO1 p_WO2 p_WO3", real=True)
    q_WO = sp.symbols("q_WO1 q_WO2 q_WO3 q_WO4", real=True)
    p_WH = sp.symbols("p_WH1 p_WH2 p_WH3", real=True)
    q_WH = sp.symbols("q_WH1 q_WH2 q_WH3 q_WH4", real=True)
    p_W_fix_point_contact_all = sp.Matrix(
        sp.symbols("p_W_fix_point_contact_all1:4", real=True)
    ).reshape(3, 1)
    p_O_fix_point_contact_all = sp.Matrix(
        sp.symbols("p_O_fix_point_contact_all1:4", real=True)
    ).reshape(3, 1)
    p_O_hand_contact_all = sp.Matrix(
        sp.symbols("p_O_hand_contact_all1:7", real=True)
    ).reshape(3, 2)
    p_H_hand_contact_all = sp.Matrix(
        sp.symbols("p_H_hand_contact_all1:7", real=True)
    ).reshape(3, 2)

    holonomic_constraint = []

    cons1 = (
        quat_on_vec(p_O_fix_point_contact_all[:, 0], q_WO)
        + sp.Matrix(p_WO)
        - p_W_fix_point_contact_all
    )
    holonomic_constraint.extend(cons1)

    for i in range(2):
        pos = (
            quat_on_vec(
                quat_on_vec(p_O_hand_contact_all[:, i], q_WO)
                + sp.Matrix(p_WO)
                - sp.Matrix(p_WH),
                quat_inv(q_WH),
            )
            - p_H_hand_contact_all[:, i]
        )
        holonomic_constraint.extend(pos)

    holonomic_constraint = sp.Matrix(holonomic_constraint)
    vars = list(p_WO) + list(q_WO) + list(p_WH) + list(q_WH)
    phi_q = holonomic_constraint.jacobian(vars)
    jac_phi_q = sp.lambdify(
        (
            p_WO,
            q_WO,
            p_WH,
            q_WH,
            p_W_fix_point_contact_all,
            p_O_fix_point_contact_all,
            p_O_hand_contact_all,
            p_H_hand_contact_all,
        ),
        phi_q,
        "numpy",
    )
    return jac_phi_q


def compute_surface_normal(
    contact_pose: np.ndarray,
    obj_state: Dict[str, np.ndarray],
    object_type: str,
    geom_size: np.ndarray | None = None,
) -> np.ndarray:
    """Compute object surface normal at contact point in world frame."""
    query_point = np.asarray(contact_pose, dtype=float).flatten()
    center_pos = np.asarray(obj_state["sliding_cube_pos"], dtype=float).reshape(3)
    quat_wxyz = np.asarray(obj_state["sliding_cube_quat"], dtype=float).reshape(4)

    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    query_local = rot.inv().apply(query_point - center_pos)

    if geom_size is None:
        if object_type == "sphere":
            geom_size = np.array([0.04])
        elif object_type in ("cylinder_short", "cylinder", "pen"):
            geom_size = np.array([0.03, 0.05])
        elif object_type == "box":
            geom_size = np.array([0.04, 0.04, 0.04])
        else:
            geom_size = np.array([0.04])

    if object_type == "sphere":
        normal_local = query_local
        norm = np.linalg.norm(normal_local)
        if norm < 1e-9:
            normal_local = np.array([1.0, 0.0, 0.0])
        else:
            normal_local = normal_local / norm

    elif object_type in ("cylinder_short", "cylinder", "pen"):
        xy = query_local[:2]
        norm_xy = np.linalg.norm(xy)
        if norm_xy < 1e-9:
            normal_local = np.array([1.0, 0.0, 0.0])
        else:
            normal_local = np.array([xy[0] / norm_xy, xy[1] / norm_xy, 0.0])

    elif object_type == "box":
        abs_local = np.abs(query_local)
        axis = int(np.argmax(abs_local))
        sign = np.sign(query_local[axis]) if abs_local[axis] > 1e-9 else 1.0
        normal_local = np.zeros(3, dtype=float)
        normal_local[axis] = sign

    else:
        normal_local = query_local
        norm = np.linalg.norm(normal_local)
        if norm < 1e-9:
            normal_local = np.array([1.0, 0.0, 0.0])
        else:
            normal_local = normal_local / norm

    normal = rot.apply(normal_local)
    return normal


def compute_E_qO_matrix(q_WO: np.ndarray) -> np.ndarray:
    w, x, y, z = q_WO[0], q_WO[1], q_WO[2], q_WO[3]
    return 0.5 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])


def get_center_state(
    state: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Treat control_left/control_right contacts as one rigid body center."""
    p_left = np.asarray(state["control_left_pos"], dtype=float).reshape(3)
    p_right = np.asarray(state["control_right_pos"], dtype=float).reshape(3)
    v_left = np.asarray(state["control_left_linvel"], dtype=float).reshape(3)
    v_right = np.asarray(state["control_right_linvel"], dtype=float).reshape(3)

    center_pos = 0.5 * (p_left + p_right)
    center_linvel = 0.5 * (v_left + v_right)

    q_hands = [state["control_left_quat"], state["control_right_quat"]]
    q_scipy_list = []
    for q_hand in q_hands:
        q_scipy = np.array([q_hand[1], q_hand[2], q_hand[3], q_hand[0]])
        q_scipy_list.append(q_scipy)

    all_rot = R.from_quat(np.stack(q_scipy_list, axis=0))
    times = np.linspace(0.0, 1.0, 2)
    slerp = Slerp(times, all_rot)

    center_rot = slerp([0.5])
    center_quat_scipy = center_rot.as_quat()[0]
    center_quat_wxyz = np.array(
        [
            center_quat_scipy[3],
            center_quat_scipy[0],
            center_quat_scipy[1],
            center_quat_scipy[2],
        ]
    )

    r_left = p_left - center_pos
    r_right = p_right - center_pos
    v_left_rel = v_left - center_linvel
    v_right_rel = v_right - center_linvel
    eps = 1e-9
    omega_left = np.cross(r_left, v_left_rel) / (np.dot(r_left, r_left) + eps)
    omega_right = np.cross(r_right, v_right_rel) / (np.dot(r_right, r_right) + eps)
    center_angvel = 0.5 * (omega_left + omega_right)

    return center_pos, center_quat_wxyz, center_linvel, center_angvel


def compute_hfvc_inputs(
    state: Dict[str, np.ndarray],
    goal_velocity: np.ndarray,
    goal_angvel: np.ndarray,
    friction_coeff_hand: float = 0.8,
    min_normal_force: float = 3.0,
    jac_phi_q_cube_rotating=None,
    object_mass: float = 0.5,
    object_type: str = "box",
    geom_size: np.ndarray | None = None,
) -> Tuple:
    """Build OCHS/HFVC optimization inputs for two moving contacts + one fixed contact."""
    if jac_phi_q_cube_rotating is None:
        raise ValueError("jac_phi_q_cube_rotating must be provided")

    kObjectMass = float(object_mass)
    kHandMass = 0.0 * 2
    kGravityConstant = 9.8
    kFrictionCoefficientHand = float(friction_coeff_hand)
    kFrictionConeSides = 6
    kMinNormalForce = float(min_normal_force)

    goal_angvel_vec = np.asarray(goal_angvel, dtype=float).reshape(3, 1)
    ang_norm = np.linalg.norm(goal_angvel_vec) + 1e-9
    kGoalAngVel = ang_norm
    kTiltDirection = goal_angvel_vec / ang_norm

    p_WO = state["sliding_cube_pos"].reshape(-1, 1)
    q_WO = state["sliding_cube_quat"].reshape(-1, 1)
    p_WH, q_WH, _, _ = get_center_state(state)
    p_WH = p_WH.reshape(-1, 1)
    q_WH = q_WH.reshape(-1, 1)

    kDimActualized = 6
    kDimUnActualized = 6
    kDimSlidingFriction = 0
    kDimLambda = 3 * 3

    R_WO = R.from_quat(q_WO.ravel(), scalar_first=True).as_matrix()
    E_qO = compute_E_qO_matrix(q_WO.ravel())
    R_WH = R.from_quat(q_WH.ravel(), scalar_first=True).as_matrix()
    E_qH = compute_E_qO_matrix(q_WH.ravel())
    omega = block_diag(R_WO, E_qO, R_WH, E_qH)

    p_W_fix_point_contact_all = state["fix_traj_pos"].reshape(3, 1)
    p_O_fix_point_contact_all = R_WO.T @ (p_W_fix_point_contact_all - p_WO)

    p_W_hand_contact_all = np.hstack(
        [
            state["control_left_pos"].reshape(3, 1),
            state["control_right_pos"].reshape(3, 1),
        ]
    )
    p_O_hand_contact_all = R_WO.T @ (p_W_hand_contact_all - p_WO)
    p_H_hand_contact_all = R_WH.T @ (p_W_hand_contact_all - p_WH)

    kRotatAxis = kTiltDirection
    goal_lin_vel = np.asarray(goal_velocity, dtype=float).reshape(3, 1)
    goal_ang_vel = kRotatAxis * kGoalAngVel
    t_WG = np.vstack([goal_lin_vel, goal_ang_vel])

    Adj_g_WO_inv = np.block(
        [
            [R_WO.T, np.zeros((3, 3))],
            [np.zeros((3, 3)), R_WO.T],
        ]
    )
    t_OG = (Adj_g_WO_inv @ t_WG).ravel()
    eye_mask = np.eye(6)
    G = np.block([eye_mask, np.zeros((6, 6))])
    b_G = t_OG.reshape(-1, 1)

    F_WGO = np.array([0, 0, -kObjectMass * kGravityConstant]).reshape(-1, 1)
    F_WGH = np.array([0, 0, -kHandMass * kGravityConstant]).reshape(-1, 1)
    F = np.vstack([R_WO.T @ F_WGO, np.zeros((3, 1)), R_WH.T @ F_WGH, np.zeros((3, 1))])

    A = np.zeros((kFrictionConeSides * 3 + 1, kDimLambda + kDimSlidingFriction))

    def friction_directions_from_normal(normal: np.ndarray, sides: int) -> np.ndarray:
        n = np.asarray(normal, dtype=float).reshape(3)
        n = n / (np.linalg.norm(n) + 1e-9)
        if abs(n[2]) < 0.9:
            t1 = np.cross(n, np.array([0, 0, 1.0]))
        else:
            t1 = np.cross(n, np.array([1.0, 0, 0]))
        t1 = t1 / (np.linalg.norm(t1) + 1e-9)
        t2 = np.cross(n, t1)
        t2 = t2 / (np.linalg.norm(t2) + 1e-9)
        dirs = np.zeros((3, sides))
        for i in range(sides):
            theta = 2 * np.pi * i / sides
            dirs[:, i] = np.cos(theta) * t1 + np.sin(theta) * t2
        return dirs

    contact_positions = [
        state["fix_traj_pos"].reshape(3, 1),
        state["control_left_pos"].reshape(3, 1),
        state["control_right_pos"].reshape(3, 1),
    ]

    for contact_idx, p_W_contact in enumerate(contact_positions):
        normal_vector = compute_surface_normal(
            p_W_contact,
            state,
            object_type,
            geom_size,
        )
        v_friction_directions = friction_directions_from_normal(
            normal_vector,
            kFrictionConeSides,
        )
        for i in range(kFrictionConeSides):
            friction_dir = v_friction_directions[:, i].reshape(-1, 1)
            constraint_vector = (
                friction_dir.T + kFrictionCoefficientHand * normal_vector.reshape(1, -1)
            )
            A[
                contact_idx * kFrictionConeSides + i,
                3 * contact_idx : 3 * contact_idx + 3,
            ] = constraint_vector.ravel()

    dir_W = (p_WH - p_W_fix_point_contact_all).reshape(3, 1)
    dir_norm = np.linalg.norm(dir_W) + 1e-9
    dir_unit = (dir_W / dir_norm).ravel()
    A[3 * kFrictionConeSides, 3:6] = dir_unit
    A[3 * kFrictionConeSides, 6:9] = dir_unit

    b_A = np.concatenate(
        [np.zeros(kFrictionConeSides * (1 + 2)), [-kMinNormalForce]]
    ).reshape(-1, 1)

    Jac_phi_q = jac_phi_q_cube_rotating(
        p_WO.ravel(),
        q_WO.ravel(),
        p_WH.ravel(),
        q_WH.ravel(),
        p_W_fix_point_contact_all.flatten(),
        p_O_fix_point_contact_all.flatten(),
        p_O_hand_contact_all.flatten(),
        p_H_hand_contact_all.flatten(),
    )

    N_ALL = Jac_phi_q @ omega
    Aeq = np.zeros((0, kDimLambda + kDimSlidingFriction))
    beq = np.array([])

    return (
        N_ALL,
        G,
        b_G,
        F,
        Aeq,
        beq,
        A,
        b_A,
        kDimActualized,
        kDimUnActualized,
        kDimSlidingFriction,
        kDimLambda,
    )
