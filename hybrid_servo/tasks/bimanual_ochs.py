from typing import Dict, Tuple

import mujoco
import numpy as np
import sympy as sp
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def generate_friction_directions(normal: np.ndarray, sides: int) -> np.ndarray:
    """Return 3xsides tangential directions uniformly spaced in the tangent plane."""
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-9)
    # pick an arbitrary perpendicular vector
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


def get_sphere_contact_normal(
    ball_center: np.ndarray,
    contact_point: np.ndarray,
) -> np.ndarray:
    normal = contact_point - ball_center
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-9:
        # Contact point is at center, use default
        return np.array([0.0, 0.0, 1.0])
    return normal / normal_norm


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


def generate_constraint_jacobian(num_hands=6):
    p_WO = sp.symbols("p_WO1 p_WO2 p_WO3", real=True)
    q_WO = sp.symbols("q_WO1 q_WO2 q_WO3 q_WO4", real=True)
    p_WH = sp.symbols("p_WH1 p_WH2 p_WH3", real=True)
    q_WH = sp.symbols("q_WH1 q_WH2 q_WH3 q_WH4", real=True)
    p_OTC_all = sp.Matrix(sp.symbols("p_OTC_all1:4", real=True)).reshape(3, 1)  # 3x1
    p_WTC_all = sp.Matrix(sp.symbols("p_WTC_all1:4", real=True)).reshape(3, 1)  # 3x1
    p_OHC_all = sp.Matrix(
        sp.symbols(f"p_OHC_all1:{3 * num_hands + 1}", real=True)
    ).reshape(3, num_hands)
    p_HHC_all = sp.Matrix(
        sp.symbols(f"p_HHC_all1:{3 * num_hands + 1}", real=True)
    ).reshape(3, num_hands)

    holonomic_constraint = []

    ## table contact
    cons1 = quat_on_vec(p_OTC_all[:, 0], q_WO) + sp.Matrix(p_WO) - p_WTC_all
    holonomic_constraint.extend(cons1)

    ## hand contact
    for i in range(num_hands):
        pos = (
            quat_on_vec(
                quat_on_vec(p_OHC_all[:, i], q_WO) + sp.Matrix(p_WO) - sp.Matrix(p_WH),
                quat_inv(q_WH),
            )
            - p_HHC_all[:, i]
        )
        holonomic_constraint.extend(pos)

    holonomic_constraint = sp.Matrix(holonomic_constraint)
    vars = list(p_WO) + list(q_WO) + list(p_WH) + list(q_WH)
    Phi_q = holonomic_constraint.jacobian(vars)
    jac_phi_q = sp.lambdify(
        (p_WO, q_WO, p_WH, q_WH, p_OTC_all, p_WTC_all, p_OHC_all, p_HHC_all),
        Phi_q,
        "numpy",
    )
    return jac_phi_q


def compute_E_qO_matrix(q_WO: np.ndarray) -> np.ndarray:
    """
    E_qO = 0.5 * [[-qx, -qy, -qz],
                  [ qw, -qz,  qy],
                  [ qz,  qw, -qx],
                  [-qy,  qx,  qw]]
    """
    w, x, y, z = q_WO[0], q_WO[1], q_WO[2], q_WO[3]

    E_qO = 0.5 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])

    return E_qO


def compute_center_quaternion_from_hands(
    q_hands: list,
) -> np.ndarray:
    """
    Compute the orientation of the virtual center using SLERP interpolation.

    Args:
        q_hands: List of hand quaternions [w, x, y, z] (scalar first)

    Returns:
        Center quaternion [w, x, y, z] (scalar first)
    """
    num_hands = len(q_hands)

    # Convert from MuJoCo convention (w,x,y,z) to scipy convention (x,y,z,w)
    q_scipy_list = []
    for q_hand in q_hands:
        q_scipy = np.array([q_hand[1], q_hand[2], q_hand[3], q_hand[0]])
        q_scipy_list.append(q_scipy)

    # Stack rotations and create Slerp interpolator
    all_rot = R.from_quat(np.stack(q_scipy_list, axis=0))
    times = np.linspace(0.0, 1.0, num_hands)
    slerp = Slerp(times, all_rot)

    # Interpolate at midpoint (t=0.5)
    center_rot = slerp([0.5])
    center_quat_scipy = center_rot.as_quat()[0]  # (x,y,z,w)

    # Convert back to MuJoCo convention (w,x,y,z)
    center_quat = np.array(
        [
            center_quat_scipy[3],  # w
            center_quat_scipy[0],  # x
            center_quat_scipy[1],  # y
            center_quat_scipy[2],  # z
        ]
    )

    return center_quat


def get_system_state(data: mujoco.MjData) -> Dict[str, np.ndarray]:
    """
    Extract system state from sensor data.

    Sensor order:
    0-2: ball position
    3-6: ball quaternion
    7-9: ball linear velocity
    10-12: ball angular velocity
    13-15: left_hand_1 position
    16-19: left_hand_1 quaternion
    20-22: left_hand_1 linear velocity
    23-25: left_hand_1 angular velocity
    26-28: left_hand_2 position
    29-32: left_hand_2 quaternion
    33-35: left_hand_2 linear velocity
    36-38: left_hand_2 angular velocity
    39-41: left_hand_3 position
    42-45: left_hand_3 quaternion
    46-48: left_hand_3 linear velocity
    49-51: left_hand_3 angular velocity
    52-54: right_hand_1 position
    55-58: right_hand_1 quaternion
    59-61: right_hand_1 linear velocity
    62-64: right_hand_1 angular velocity
    65-67: right_hand_2 position
    68-71: right_hand_2 quaternion
    72-74: right_hand_2 linear velocity
    75-77: right_hand_2 angular velocity
    78-80: right_hand_3 position
    81-84: right_hand_3 quaternion
    85-87: right_hand_3 linear velocity
    88-90: right_hand_3 angular velocity
    91-93: left_hand_1 force
    94-96: left_hand_1 torque
    97-99: left_hand_2 force
    100-102: left_hand_2 torque
    103-105: left_hand_3 force
    106-108: left_hand_3 torque
    109-111: right_hand_1 force
    112-114: right_hand_1 torque
    115-117: right_hand_2 force
    118-120: right_hand_2 torque
    121-123: right_hand_3 force
    124-126: right_hand_3 torque
    """
    state = {
        "ball_pos": data.sensordata[0:3].copy(),
        "ball_quat": data.sensordata[3:7].copy(),
        "ball_linvel": data.sensordata[7:10].copy(),
        "ball_angvel": data.sensordata[10:13].copy(),
        "left_hand_1_pos": data.sensordata[13:16].copy(),
        "left_hand_1_quat": data.sensordata[16:20].copy(),
        "left_hand_1_linvel": data.sensordata[20:23].copy(),
        "left_hand_1_angvel": data.sensordata[23:26].copy(),
        "left_hand_2_pos": data.sensordata[26:29].copy(),
        "left_hand_2_quat": data.sensordata[29:33].copy(),
        "left_hand_2_linvel": data.sensordata[33:36].copy(),
        "left_hand_2_angvel": data.sensordata[36:39].copy(),
        "left_hand_3_pos": data.sensordata[39:42].copy(),
        "left_hand_3_quat": data.sensordata[42:46].copy(),
        "left_hand_3_linvel": data.sensordata[46:49].copy(),
        "left_hand_3_angvel": data.sensordata[49:52].copy(),
        "right_hand_1_pos": data.sensordata[52:55].copy(),
        "right_hand_1_quat": data.sensordata[55:59].copy(),
        "right_hand_1_linvel": data.sensordata[59:62].copy(),
        "right_hand_1_angvel": data.sensordata[62:65].copy(),
        "right_hand_2_pos": data.sensordata[65:68].copy(),
        "right_hand_2_quat": data.sensordata[68:72].copy(),
        "right_hand_2_linvel": data.sensordata[72:75].copy(),
        "right_hand_2_angvel": data.sensordata[75:78].copy(),
        "right_hand_3_pos": data.sensordata[78:81].copy(),
        "right_hand_3_quat": data.sensordata[81:85].copy(),
        "right_hand_3_linvel": data.sensordata[85:88].copy(),
        "right_hand_3_angvel": data.sensordata[88:91].copy(),
        "left_hand_1_force": data.sensordata[91:94].copy(),
        "left_hand_1_torque": data.sensordata[94:97].copy(),
        "left_hand_2_force": data.sensordata[97:100].copy(),
        "left_hand_2_torque": data.sensordata[100:103].copy(),
        "left_hand_3_force": data.sensordata[103:106].copy(),
        "left_hand_3_torque": data.sensordata[106:109].copy(),
        "right_hand_1_force": data.sensordata[109:112].copy(),
        "right_hand_1_torque": data.sensordata[112:115].copy(),
        "right_hand_2_force": data.sensordata[115:118].copy(),
        "right_hand_2_torque": data.sensordata[118:121].copy(),
        "right_hand_3_force": data.sensordata[121:124].copy(),
        "right_hand_3_torque": data.sensordata[124:127].copy(),
    }
    return state


def get_contact_point(model, data, geom_name):
    """Get contact point for a specific geom."""
    if data.ncon > 0:
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            )
            geom2_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            )
            if (geom1_name == geom_name and geom2_name == "rolling_ball_geom") or (
                geom1_name == "rolling_ball_geom" and geom2_name == geom_name
            ):
                return contact.pos
    return None


def compute_ochs_inputs(
    state: Dict[str, np.ndarray],
    goal_angular_velocity: float = 0.5,
    goal_rotate_axis: np.ndarray = np.array([0, 1, 0]),
    friction_coeff_ground: float = 0.8,
    friction_coeff_hand: float = 0.8,
    kMinHandNormalForce: float = 10.0,
    active_hands: str = "both",  # "left", "right", or "both"
    jacobian=None,
    kBallMass=1.5,
    kBallRadius=0.08,
) -> Tuple:
    # Determine which hands are active
    if active_hands == "left":
        active_hand_indices = [0, 1, 2]  # left_hand_1, left_hand_2, left_hand_3
    elif active_hands == "right":
        active_hand_indices = [3, 4, 5]  # right_hand_1, right_hand_2, right_hand_3
    elif active_hands == "both":
        active_hand_indices = [0, 1, 2, 3, 4, 5]  # all 6 hands
    else:
        raise ValueError(
            f"Invalid active_hands value: {active_hands}. Must be 'left', 'right', or 'both'"
        )

    num_active_hands = len(active_hand_indices)

    kHandMass = 0.00 * num_active_hands  # Active hands total mass
    kGravityConstant = 9.8
    kDimActualized = 6  # Virtual rigid body: 3 linear + 2 angular
    kDimUnActualized = 6  # Ball: 3 linear + 3 angular
    kDimSlidingFriction = 0

    kFrictionCoefficientGround = friction_coeff_ground
    kFrictionConeSides = 6

    kGoalAngularVelocity = goal_angular_velocity
    kRotateAxis = goal_rotate_axis.reshape(-1, 1)

    # Ball state
    p_WO = state["ball_pos"].reshape(-1, 1)
    q_WO = state["ball_quat"].reshape(-1, 1)

    # All hand state keys (global indexing 0-5)
    hand_pos_keys = [
        "left_hand_1_pos",
        "left_hand_2_pos",
        "left_hand_3_pos",
        "right_hand_1_pos",
        "right_hand_2_pos",
        "right_hand_3_pos",
    ]
    hand_quat_keys = [
        "left_hand_1_quat",
        "left_hand_2_quat",
        "left_hand_3_quat",
        "right_hand_1_quat",
        "right_hand_2_quat",
        "right_hand_3_quat",
    ]
    # Virtual rigid body center position (average of active hands)
    hand_positions = [state[hand_pos_keys[i]] for i in active_hand_indices]
    p_WH = (sum(hand_positions) / num_active_hands).reshape(-1, 1)
    # Virtual rigid body orientation - compute from active hands using SLERP
    hand_quats = [state[hand_quat_keys[i]] for i in active_hand_indices]
    q_WH = hand_quats[0].reshape(-1, 1)
    # Rotation matrices
    R_WO = R.from_quat(q_WO.ravel(), scalar_first=True).as_matrix()
    R_WH = R.from_quat(q_WH.ravel(), scalar_first=True).as_matrix()

    # E matrices
    E_qO = compute_E_qO_matrix(q_WO.ravel())
    E_qH = compute_E_qO_matrix(q_WH.ravel())

    # Omega matrix
    Omega = block_diag(R_WO, E_qO, R_WH, E_qH)  # (14, 12)

    # Goal: ball rotates around specified axis
    omega_WG = kRotateAxis * kGoalAngularVelocity
    linvel_WG = -np.cross(
        omega_WG.flatten(), -kBallRadius * np.array([0, 0, 1])
    ).reshape(-1, 1)
    t_WG = np.vstack([linvel_WG, omega_WG])
    Adj_g_WO_inv = np.block(
        [
            [R_WO.T, np.zeros((3, 3))],
            [np.zeros((3, 3)), R_WO.T],
        ]
    )
    t_OG = (Adj_g_WO_inv @ t_WG).ravel()
    G = np.block([np.eye(6), np.zeros((6, 6))])  # (6, 12)
    b_G = t_OG.reshape(-1, 1)  # (6, 1)

    F_WGO = np.array([0, 0, -kBallMass * kGravityConstant]).reshape(-1, 1)
    F_WGH = np.array([0, 0, -kHandMass * kGravityConstant]).reshape(-1, 1)

    F = np.vstack(
        [
            R_WO.T @ F_WGO,  # Ball gravity in ball frame
            np.zeros((3, 1)),  # No external torque on ball
            F_WGH,  # Rigid body gravity
            np.zeros((3, 1)),
        ]
    )
    kDimLambda = 3 + 3 * num_active_hands

    A = np.zeros((kFrictionConeSides * (1 + num_active_hands) + 1, kDimLambda))

    ground_normal = np.array([0, 0, -1])
    ground_friction_dirs = generate_friction_directions(
        ground_normal, kFrictionConeSides
    )
    for i in range(kFrictionConeSides):
        friction_dir = ground_friction_dirs[:, i].reshape(-1, 1)
        constraint_vector = (
            friction_dir.T + kFrictionCoefficientGround * ground_normal.reshape(1, -1)
        )
        A[i, 0:3] = constraint_vector.ravel()

    for local_idx in range(num_active_hands):
        hand_pos = hand_positions[local_idx].reshape(3)
        contact_normal = get_sphere_contact_normal(p_WO.flatten(), hand_pos)
        hand_friction_dirs = generate_friction_directions(
            contact_normal, kFrictionConeSides
        )

        for i in range(kFrictionConeSides):
            friction_dir = hand_friction_dirs[:, i].reshape(-1, 1)
            constraint_vector = (
                friction_dir.T + friction_coeff_hand * contact_normal.reshape(1, -1)
            )
            row_idx = kFrictionConeSides * (local_idx + 1) + i
            col_start = 3 * (local_idx + 1)
            A[row_idx, col_start : col_start + 3] = constraint_vector.ravel()

    sum_row_idx = kFrictionConeSides * (1 + num_active_hands)

    p_W_ground_contact = p_WO.copy()
    p_W_ground_contact[2] = 0.0

    dir_W = (p_WH - p_W_ground_contact).reshape(3, 1)
    dir_norm = np.linalg.norm(dir_W) + 1e-9
    dir_unit = (dir_W / dir_norm).ravel()

    for idx in range(num_active_hands):
        col_start = 3 * (idx + 1)
        A[sum_row_idx, col_start : col_start + 3] = dir_unit

    b_A = np.concatenate(
        [
            np.zeros(kFrictionConeSides * (1 + num_active_hands)),
            [-kMinHandNormalForce],
        ]
    ).reshape(-1, 1)

    p_WHC_all = []
    for i in range(num_active_hands):
        p_WHC_all.append(hand_positions[i].reshape(-1, 1))
    p_OHC_list = [R_WO.T @ (p_WHC - p_WO) for p_WHC in p_WHC_all]
    p_HHC_list = [R_WH.T @ (p_WHC - p_WH) for p_WHC in p_WHC_all]

    p_OHC_all = np.hstack(p_OHC_list)
    p_HHC_all = np.hstack(p_HHC_list)
    p_OTC_all = np.array([0, 0, -kBallRadius]).reshape(-1, 1)
    p_WTC_all = p_WO + p_OTC_all
    Jac_phi_q = jacobian(
        p_WO.ravel(),
        q_WO.ravel(),
        p_WH.ravel(),
        q_WH.ravel(),
        p_OTC_all.flatten(),
        p_WTC_all.flatten(),
        p_OHC_all.flatten(),
        p_HHC_all.flatten(),
    )

    N_ALL = Jac_phi_q @ Omega
    Aeq = np.zeros((0, kDimLambda))
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
