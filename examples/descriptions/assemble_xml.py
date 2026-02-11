"""Robot XML assembly module for MuJoCo simulation.

Assembles complete robot XML descriptions by combining torso, arm, and leg components
from OnShape CAD exports. Handles collision mesh processing, joint constraints,
keyframe generation, and output formatting for various simulation environments.
"""

import argparse
import os
import re
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import Dict, List

import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from toddlerbot.utils.io_utils import pretty_write_xml
from toddlerbot.utils.math_utils import round_to_sig_digits


def string_to_list(string: str) -> list:
    """Convert a comma-separated string to a list of strings."""
    return [float(item.strip()) for item in string.split(" ") if item.strip()]


def list_to_string(lst: list, digits: int = 8) -> str:
    """Convert a list of strings to a comma-separated string."""
    return " ".join(str(round(item, digits)) for item in lst)


def compute_bounding_box(mesh: trimesh.Trimesh):
    """Computes the size and center of the bounding box for a given 3D mesh.

    Args:
        mesh (trimesh.Trimesh): The 3D mesh for which to compute the bounding box.

    Returns:
        tuple: A tuple containing the size (width, height, depth) of the bounding box and the center point of the bounding box.
    """
    # Compute the minimum and maximum bounds along each axis
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]

    # Calculate the size (width, height, depth) of the bounding box
    size = (bounds_max - bounds_min) / 2

    # The center of the bounding box
    center = (bounds_max + bounds_min) / 2

    return size, center


def update_collision(
    old_geom: ET.Element,
    collision_config: dict,
    mesh_dir: str,
    ignore_keywords: list = [
        "arm",
        "leg",
        "gear",
        "neck",
        "hip_roll",
        "shoulder_pitch",
    ],
):
    """Updates MuJoCo geom element to minimal-volume bounding primitive or convex mesh.

    Args:
        old_geom: Original geometry element to replace
        collision_config: Configuration mapping mesh names to collision shapes
        mesh_dir: Directory containing mesh files
        ignore_keywords: Keywords to skip collision generation for

    Returns:
        List of new geometry elements for collision
    """
    mesh_name = old_geom.get("mesh")
    assert mesh_name, "Mesh filename must be provided in the geometry element."

    for key in ignore_keywords:
        if key in mesh_name:
            return []

    geom_key = ""
    for key in collision_config:
        if key in mesh_name:
            geom_key = key
            break

    def get_box_geom(pos, size, i=None):
        return ET.Element(
            "geom",
            {
                "type": "box",
                "class": "collision",
                "pos": list_to_string(pos),
                "quat": old_geom.attrib["quat"],
                "size": list_to_string(size),
                "name": mesh_name if i is None else mesh_name + f"_{i}",
                "material": old_geom.attrib["material"].replace("collision", "visual"),
            },
        )

    def get_convex_hull(max_vertices=28, cutoff_height=None):
        def layered_fps(points, k, z_tol=1e-4):
            """
            Perform farthest point sampling on the two dominant Z layers of the input points.

            Args:
                points: (N, 3) array of 3D points
                k: total number of points to sample (divided roughly equally between layers)
                z_tol: tolerance to group Z-values into the same layer

            Returns:
                (k, 3) array of sampled points
            """
            z_values = points[:, 2]
            # Round Z values to cluster similar layers
            rounded_z = np.round(z_values / z_tol) * z_tol

            # Find the two most common Z values
            unique, counts = np.unique(rounded_z, return_counts=True)
            sorted_indices = np.argsort(-counts)
            z1, z2 = unique[sorted_indices[:2]]

            # Split points by layer
            layer1 = points[np.abs(rounded_z - z1) < z_tol]
            layer2 = points[np.abs(rounded_z - z2) < z_tol]

            # Allocate samples proportionally or evenly
            k1 = k // 2
            k2 = k - k1

            np.random.seed(0)  # For reproducibility

            def fps(pts, num):
                if len(pts) <= num:
                    return pts
                selected = [np.random.randint(len(pts))]
                for _ in range(1, num):
                    dists = cdist(pts[selected], pts)
                    min_dists = np.min(dists, axis=0)
                    next_index = np.argmax(min_dists)
                    selected.append(next_index)
                return pts[selected]

            sampled1 = fps(layer1, min(k1, len(layer1)))
            sampled2 = fps(layer2, min(k2, len(layer2)))

            return np.vstack([sampled1, sampled2])

        mesh_path = os.path.join(mesh_dir, f"{mesh_name}.stl")
        mesh = trimesh.load(mesh_path)
        vertices = mesh.vertices
        if cutoff_height:
            vertices = vertices[vertices[:, 2] <= cutoff_height]

        if len(vertices) < 4:
            raise ValueError("Not enough points below cutoff to form a convex hull.")

        if len(vertices) > max_vertices:
            sampled_points = layered_fps(vertices, max_vertices)
        else:
            sampled_points = vertices

        # Mirror across Y-axis (flip X) to enforce symmetry
        mirrored = sampled_points.copy()
        mirrored[:, 1] *= -1
        symmetric_points = np.vstack([sampled_points, mirrored])

        hull = ConvexHull(symmetric_points)
        faces = hull.simplices

        convex_mesh = trimesh.Trimesh(
            vertices=symmetric_points, faces=faces, process=True
        )

        # Ensure correct normals (sometimes they're flipped)
        convex_mesh.fix_normals()
        convex_mesh.remove_unreferenced_vertices()
        convex_mesh = convex_mesh.convex_hull  # ensure watertight

        convex_mesh.export(mesh_path)

        print(
            f"Exported symmetric convex hull with {len(convex_mesh.vertices)} vertices to {mesh_path}"
        )

    def get_mesh_geom():
        # The right one is copied
        if "left_ankle_roll_link" in mesh_name:
            # TODO: Hardcoded for left ankle roll link
            get_convex_hull(cutoff_height=0.003255)

        return ET.Element(
            "geom",
            {
                "type": "mesh",
                "class": "collision",
                "mesh": mesh_name,
                "pos": old_geom.attrib["pos"],
                "quat": old_geom.attrib["quat"],
                "name": mesh_name,
                "material": old_geom.attrib["material"].replace("collision", "visual"),
                "contype": "1",
                "conaffinity": "1",
            },
        )

    new_geom_list = []
    if geom_key:
        if (
            "type" in collision_config[geom_key]
            and collision_config[geom_key]["type"] == "mesh"
        ):
            new_geom_list.append(get_mesh_geom())
        elif isinstance(collision_config[geom_key]["pos"][0], list):
            for i, (pos, size) in enumerate(
                zip(
                    collision_config[geom_key]["pos"],
                    collision_config[geom_key]["size"],
                )
            ):
                new_geom_list.append(get_box_geom(pos, size, i))
        else:
            new_geom_list.append(
                get_box_geom(
                    collision_config[geom_key]["pos"],
                    collision_config[geom_key]["size"],
                )
            )

    else:
        mesh = trimesh.load(os.path.join(mesh_dir, f"{mesh_name}.stl"))

        box_size, box_center = compute_bounding_box(mesh)
        R_geom = R.from_quat(string_to_list(old_geom.attrib["quat"]), scalar_first=True)
        new_geom = get_box_geom(
            np.array(string_to_list(old_geom.attrib["pos"])) + R_geom.apply(box_center),
            box_size,
        )
        # print(
        #     f"{mesh_name}: size={new_geom.attrib['size']}, pos={new_geom.attrib['pos']}"
        # )
        new_geom_list.append(new_geom)

    return new_geom_list


def find_body_and_parent(root: ET.Element, target_name: str):
    """Recursively find the body with name=target_name and its parent."""
    for child in root:
        if child.tag == "body":
            if child.attrib.get("name") == target_name:
                return root, child  # Found: return parent and target
            result = find_body_and_parent(child, target_name)
            if result:
                return result
    return None


def align_body_frames_to_global(root):
    """Aligns all body frames in the MJCF tree to the global frame (quat = 1 0 0 0)
    and updates all child elements (joint axis, geom/site/inertial pos and quat)
    accordingly so the world behavior is preserved.
    """

    def update_child_frames(body, parent_global_rot):
        body_pos = string_to_list(body.get("pos", "0 0 0"))
        body_rot = R.from_quat(
            string_to_list(body.get("quat", "1 0 0 0")), scalar_first=True
        )
        global_rot = parent_global_rot * body_rot
        global_pos = parent_global_rot.apply(body_pos)
        # Now align body to global
        body.set("quat", "1 0 0 0")
        body.set("pos", list_to_string(global_pos))
        for child in body:
            if child.tag in ["geom", "site", "camera", "light", "inertial"]:
                if "pos" in child.attrib:
                    old_pos = string_to_list(child.attrib["pos"])
                    new_pos = global_rot.apply(old_pos)
                    child.set("pos", list_to_string(new_pos))
                if "quat" in child.attrib:
                    old_quat = R.from_quat(
                        string_to_list(child.attrib["quat"]), scalar_first=True
                    )
                    new_quat = global_rot * old_quat
                    child.set(
                        "quat", list_to_string(new_quat.as_quat(scalar_first=True))
                    )
            elif child.tag == "joint":
                if "axis" in child.attrib:
                    axis = string_to_list(child.attrib["axis"])
                    new_axis = global_rot.apply(axis)
                    new_axis = [round(a, 2) for a in new_axis]
                    child.set("axis", list_to_string(new_axis))
            elif child.tag == "body":
                update_child_frames(child, global_rot)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("<worldbody> element not found")

    for body in worldbody.findall("body"):
        update_child_frames(body, R.identity())

    return root


def add_waist_constraints(root: ET.Element, offsets: Dict[str, float]):
    """Adds waist constraints to the given XML element by creating and configuring tendon elements for waist roll and yaw.

    This function modifies the provided XML element by removing any existing 'tendon' element and adding a new one with specific constraints for waist roll and yaw. The constraints are defined using coefficients and backlash values from the general configuration.

    Args:
        root (ET.Element): The root XML element to which the waist constraints will be added.
        general_config (Dict[str, Any]): A dictionary containing configuration values, including offsets and backlash for waist roll and yaw.
    """
    # Ensure there is an <equality> element
    tendon = ET.SubElement(root, "tendon")

    waist_roll_coef = offsets["waist_roll_coef"]
    waist_yaw_coef = offsets["waist_yaw_coef"]

    waist_roll_backlash = offsets["waist_roll_backlash"]
    waist_yaw_backlash = offsets["waist_yaw_backlash"]

    # waist roll
    fixed_roll = ET.SubElement(
        tendon,
        "fixed",
        name="waist_roll_coupling",
        limited="true",
        range=f"-{waist_roll_backlash} {waist_roll_backlash}",
    )
    ET.SubElement(fixed_roll, "joint", joint="waist_act_1", coef=f"{waist_roll_coef}")
    ET.SubElement(fixed_roll, "joint", joint="waist_act_2", coef=f"{-waist_roll_coef}")
    ET.SubElement(fixed_roll, "joint", joint="waist_roll", coef="1")

    # waist roll
    fixed_yaw = ET.SubElement(
        tendon,
        "fixed",
        name="waist_yaw_coupling",
        limited="true",
        range=f"-{waist_yaw_backlash} {waist_yaw_backlash}",
    )
    ET.SubElement(fixed_yaw, "joint", joint="waist_act_1", coef=f"{-waist_yaw_coef}")
    ET.SubElement(fixed_yaw, "joint", joint="waist_act_2", coef=f"{-waist_yaw_coef}")
    ET.SubElement(fixed_yaw, "joint", joint="waist_yaw", coef="1")


def add_gripper_constraints(root: ET.Element, offsets: Dict[str, float]):
    equality = root.find(".//equality")

    for side in ["left", "right"]:
        joint_name = f"{side}_gripper_rack"
        joint_pinion_1_name = f"{side}_gripper_pinion"
        joint_pinion_2_name = f"{side}_gripper_pinion_mirror"
        for joint_pinion_name in [joint_pinion_1_name, joint_pinion_2_name]:
            joint_pinion: ET.Element | None = root.find(
                f".//joint[@name='{joint_pinion_name}']"
            )
            if joint_pinion is None:
                raise ValueError(f"The pinion joint {joint_pinion_name} is not found")

            ET.SubElement(
                equality,
                "joint",
                joint1=joint_pinion_name,
                joint2=joint_name,
                polycoef=f"0 {offsets['gripper_gear_ratio']} 0 0 0",
                solimp="0.9999 0.9999 0.001 0.5 2",
                solref="0.004 1",
            )


def update_joint_params(root: ET.Element, joint_params: Dict[str, float]):
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name is None:
            continue
        match = next(
            (params for key, params in joint_params.items() if key in name), None
        )
        if match:
            for attr, value in match.items():
                joint.set(attr, str(value))


def add_sites(
    root: ET.Element, offsets: Dict[str, float], robot_config: Dict[str, str]
):
    """Adds foot sites to the XML structure based on the specified foot name in the configuration.

    Args:
        root (ET.Element): The root element of the XML structure to which foot sites will be added.
        foot_name (str): The name of the foot to identify target geometries.

    The function searches for geometries in the XML structure that match the specified foot name and contain "collision" in their name. For each matching geometry, it calculates the position for a new site and adds it as a child element to the geometry's parent, with predefined specifications for type, size, and color.
    """
    # Site specifications
    site_specifications = {"type": "sphere", "size": "0.005", "rgba": "0.9 0.1 0.1 0.8"}

    torso_center_pos = offsets.get("torso_site_pos")
    if torso_center_pos:
        torso = root.find(".//body[@name='torso']")
        if torso is not None and torso.find("site[@name='torso_center']") is None:
            ET.SubElement(
                torso,
                "site",
                {
                    "name": "torso_center",
                    "pos": list_to_string(torso_center_pos),
                    **site_specifications,
                },
            )

    if "hand_name" in robot_config:
        hand_name = robot_config["hand_name"]
        hand_geom_dict = {"left": [], "right": []}
        for parent in root.iter():
            for geom in parent.findall("geom"):
                name = geom.attrib.get("name", "")
                if "collision" in name and hand_name in name:
                    side = "left" if "left" in name else "right"
                    hand_geom_dict[side].append((parent, geom))

        for side, value in hand_geom_dict.items():
            parent, geom = value[0]
            geom_pos = string_to_list(geom.attrib["pos"])
            geom_size = string_to_list(geom.attrib["size"])

            if "gripper" in hand_name:
                bottom_center_pos = [
                    geom_pos[0],
                    geom_pos[1] + geom_size[2]
                    if geom_pos[1] > 0
                    else geom_pos[1] - geom_size[2],
                    geom_pos[2],
                ]
            elif "leader" in hand_name:
                bottom_center_pos = [
                    geom_pos[0],
                    geom_pos[1] + geom_size[2]
                    if geom_pos[1] > 0
                    else geom_pos[1] - geom_size[2],
                    geom_pos[2],
                ]
            else:
                bottom_center_pos = [
                    geom_pos[0],
                    geom_pos[1] + geom_size[2]
                    if geom_pos[1] > 0
                    else geom_pos[1] - geom_size[2],
                    geom_pos[2],
                ]

            ET.SubElement(
                parent,
                "site",
                {
                    "name": f"{side}_hand_center",
                    "pos": list_to_string(bottom_center_pos),
                    **site_specifications,
                },
            )

    if "foot_name" in robot_config:
        foot_name = robot_config["foot_name"]
        foot_geom_dict = {"left": [], "right": []}
        for parent in root.iter():
            for geom in parent.findall("geom"):
                name = geom.attrib.get("name", "")
                if "collision" in name and foot_name in name:
                    side = "left" if "left" in name else "right"
                    foot_geom_dict[side].append((parent, geom))

        for side, value in foot_geom_dict.items():
            parent, geom = value[0]
            geom_pos = string_to_list(geom.attrib["pos"])
            geom_size = string_to_list(geom.attrib["size"])
            bottom_center_pos = [
                geom_pos[0] - min(geom_size),
                geom_pos[1],
                geom_pos[2],
            ]

            ET.SubElement(
                parent,
                "site",
                {
                    "name": f"{side}_foot_center",
                    "pos": list_to_string(bottom_center_pos),
                    **site_specifications,
                },
            )


def add_site_velocity_sensors(root: ET.Element, site_names: List[str]):
    """Ensures linear and angular velocity sensors exist for the provided sites.

    Args:
        root (ET.Element): The XML root element that may contain a <sensor> section.
        site_names (List[str]): Names of sites that should have paired velocity sensors.

    The function creates a <sensor> block if needed and adds framelinvel/frameangvel sensors
    for each existing site in the provided list, skipping any sensors that already exist.
    """
    sensor_elem = root.find("sensor")
    if sensor_elem is None:
        sensor_elem = ET.SubElement(root, "sensor")

    existing_names = {child.get("name") for child in sensor_elem}

    for site_name in site_names:
        site = root.find(f".//site[@name='{site_name}']")
        if site is None:
            continue

        linvel_name = f"{site_name}_linvel"
        if linvel_name not in existing_names:
            ET.SubElement(
                sensor_elem,
                "framelinvel",
                {"objtype": "site", "objname": site_name, "name": linvel_name},
            )
            existing_names.add(linvel_name)

        angvel_name = f"{site_name}_angvel"
        if angvel_name not in existing_names:
            ET.SubElement(
                sensor_elem,
                "frameangvel",
                {"objtype": "site", "objname": site_name, "name": angvel_name},
            )
            existing_names.add(angvel_name)


def add_self_contacts(root: ET.Element, contact_config: Dict[str, List[str]]):
    """Adds contact and exclusion pairs to an XML element based on a collision configuration.

    Args:
        root (ET.Element): The root XML element to which contact pairs and exclusions will be added.
        collision_config (Dict[str, Dict[str, Any]]): A dictionary containing collision configuration for each body. Each entry specifies which other bodies it can contact with.

    Raises:
        ValueError: If a geometry name cannot be found for any of the specified body pairs.
    """
    contact = ET.SubElement(root, "contact")
    collision_geoms = root.findall(".//geom[@class='collision']")

    # Build mapping from body name to (geom_elem, geom_name)
    geom_map = {}
    for geom in collision_geoms:
        geom_name = geom.get("name")
        if geom_name is not None:
            geom_map[geom_name] = geom

    geom_names = list(geom_map.keys())

    for geom1_name in geom_names:
        geom1_keywords = []
        for key in contact_config:
            if key in geom1_name:
                geom1_keywords = contact_config[key]
                break

        for geom2_name in geom_names:
            if geom1_name == geom2_name:
                continue

            # Check if any keyword from contact_config[geom1_name] is in geom2_name
            if any(keyword in geom2_name for keyword in geom1_keywords):
                ET.SubElement(contact, "pair", geom1=geom1_name, geom2=geom2_name)


def add_keyframes(
    root: ET.Element,
    home_pos: Dict[str, float],
    offsets: Dict[str, float],
    target_robot_dir: str,
):
    keyframe_elem = ET.SubElement(root, "keyframe")

    robot_name = os.path.basename(target_robot_dir)
    robot_xml_path = os.path.join(target_robot_dir, f"{robot_name}_temp.xml")
    pretty_write_xml(root, robot_xml_path)
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    data = mujoco.MjData(model)

    def get_transmission(motor_name: str) -> str:
        if motor_name.endswith("_drive"):
            return "spur_gear"
        elif motor_name.endswith("_act"):
            return "parallel_linkage"
        elif re.search(r"_act_[12]$", motor_name):
            return "bevel_gear"
        elif motor_name.endswith("_rack"):
            return "rack_and_pinion"
        else:
            return "none"

    dof_angles: Dict[str, float] = {}
    dof_angles.update(home_pos)

    waist_act_1_pos = None
    for motor_name, motor_pos in home_pos.items():
        transmission = get_transmission(motor_name)
        if transmission == "spur_gear":
            joint_name = motor_name.replace("_drive", "_driven")
            gear_ratio = float(
                root.find(f".//equality/joint[@joint1='{joint_name}']")
                .attrib["polycoef"]
                .split()[1]
            )
            dof_angles[joint_name] = motor_pos * gear_ratio
        elif transmission == "parallel_linkage":
            joint_name = motor_name.replace("_act", "")
            dof_angles[joint_name] = motor_pos
            for suffix in ["_front", "_back"]:
                dof_angles[joint_name + suffix] = -motor_pos
        elif transmission == "bevel_gear":
            # Placeholder to ensure the correct order
            if waist_act_1_pos is None:
                waist_act_1_pos = motor_pos
                continue
            else:
                dof_angles["waist_roll"] = offsets["waist_roll_coef"] * (
                    -waist_act_1_pos + motor_pos
                )
                dof_angles["waist_yaw"] = offsets["waist_yaw_coef"] * (
                    waist_act_1_pos + motor_pos
                )
        elif transmission == "rack_and_pinion":
            joint_pinion_name = motor_name.replace("_rack", "_pinion")
            joint_pos = -motor_pos * offsets["gripper_gear_ratio"]
            dof_angles[joint_pinion_name] = joint_pos
            dof_angles[joint_pinion_name + "_mirror"] = joint_pos
        elif transmission == "none":
            dof_angles[motor_name] = motor_pos

    mujoco.mj_forward(model, data)
    com_0 = data.body(0).subtree_com.copy()

    # for name, dof_pos in dof_angles.items():
    #     if "hip" in name or "knee" in name or "ankle" in name:
    #         continue

    #     data.joint(name).qpos = dof_pos

    # mujoco.mj_forward(model, data)
    # com_1 = data.body(0).subtree_com.copy()

    # # Code to calculate the leg joint angles to move the CoM to the center of the support polygon
    # hip_pitch_id = mujoco.mj_name2id(
    #     model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link"
    # )
    # knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")
    # ankle_pitch_id = mujoco.mj_name2id(
    #     model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_pitch_link"
    # )

    # hip_to_knee = data.xpos[hip_pitch_id] - data.xpos[knee_id]
    # hip_to_knee[1] = 0.0
    # knee_to_ankle = data.xpos[knee_id] - data.xpos[ankle_pitch_id]
    # knee_to_ankle[1] = 0.0
    # hip_to_ankle = data.xpos[hip_pitch_id] - data.xpos[ankle_pitch_id]
    # hip_to_ankle[1] = 0.0
    # hip_to_knee_len = np.linalg.norm(hip_to_knee)
    # knee_to_ankle_len = np.linalg.norm(knee_to_ankle)

    # hip_to_ankle_target = hip_to_ankle.copy()
    # hip_to_ankle_target[0] -= com_1[0]
    # hip_to_ankle_target[2] += offsets["home_pos_z_delta"]
    # knee_cos = (
    #     hip_to_knee_len**2
    #     + knee_to_ankle_len**2
    #     - np.linalg.norm(hip_to_ankle_target) ** 2
    # ) / (2 * hip_to_knee_len * knee_to_ankle_len)
    # knee_cos = np.clip(knee_cos, -1.0, 1.0)
    # knee = np.abs(np.pi - np.arccos(knee_cos))

    # ank_pitch = np.arctan2(
    #     np.sin(knee), np.cos(knee) + knee_to_ankle_len / hip_to_knee_len
    # ) + np.arctan2(hip_to_ankle_target[0], hip_to_ankle_target[2])
    # hip_pitch = knee - ank_pitch
    # print(
    #     f"Calculated leg angles: hip_pitch={hip_pitch:.6f}, knee={knee:.6f}, ankle_pitch={ank_pitch:.6f}"
    # )

    for name, dof_pos in dof_angles.items():
        data.joint(name).qpos = dof_pos

    # mujoco.mj_forward(model, data)
    # com_2 = data.body(0).subtree_com.copy()

    if waist_act_1_pos is not None:
        root_pos = np.array(offsets["zero_pos"])
        root_pos[0] -= com_0[0]
        root_pos[2] += offsets["home_pos_z_delta"]
        data.qpos[:3] = root_pos

    # mujoco.mj_forward(model, data)
    # com_3 = data.body(0).subtree_com.copy()

    ET.SubElement(
        keyframe_elem,
        "key",
        {
            "name": "home",
            "qpos": f"{list_to_string(data.qpos)}",
            "ctrl": list_to_string(home_pos.values()),
        },
    )

    # Clean up the temporary XML file
    if os.path.exists(robot_xml_path):
        os.remove(robot_xml_path)


def add_offsets(root: ET.Element, robot_config: dict, target_robot_dir: str):
    """Adds offsets to the robot configuration based on the provided local configuration.

    Args:
        robot_config_local (Dict[str, Dict[str, float]]): A dictionary containing the robot's local configuration with offsets.
        root (ET.Element): The root XML element to which the offsets will be added.

    This function iterates through the robot configuration and adds an 'offset' element for each key in the local configuration.
    """
    robot_name = os.path.basename(target_robot_dir)
    robot_xml_path = os.path.join(target_robot_dir, f"{robot_name}_temp.xml")
    pretty_write_xml(root, robot_xml_path)
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    com = data.body(0).subtree_com.copy()

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    hip_pitch_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link"
    )
    hip_roll_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_hip_roll_link"
    )
    knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")
    ankle_pitch_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_pitch_link"
    )
    ankle_roll_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
    )

    hip_pitch_to_roll_z = abs(
        (data.xpos[hip_pitch_id] - data.xpos[hip_roll_id])[2].item()
    )
    torso_to_hip_z = abs((data.xpos[torso_id] - data.xpos[hip_pitch_id])[2].item())
    hip_to_knee_z = abs((data.xpos[hip_pitch_id] - data.xpos[knee_id])[2].item())
    knee_to_ankle_z = abs((data.xpos[knee_id] - data.xpos[ankle_pitch_id])[2].item())
    hip_to_ankle_pitch_z = abs(
        (data.xpos[hip_pitch_id] - data.xpos[ankle_pitch_id])[2].item()
    )
    hip_to_ankle_roll_z = abs(
        (data.xpos[hip_roll_id] - data.xpos[ankle_roll_id])[2].item()
    )
    foot_to_com_y = abs(data.xpos[ankle_roll_id][1].item())

    robot_config["com_x"] = round_to_sig_digits(com[0].item(), 4)
    robot_config["com_z"] = round_to_sig_digits(com[2].item(), 4)
    robot_config["hip_pitch_to_roll_z"] = round_to_sig_digits(hip_pitch_to_roll_z, 4)
    robot_config["torso_to_hip_z"] = round_to_sig_digits(torso_to_hip_z, 4)
    robot_config["hip_to_knee_z"] = round_to_sig_digits(hip_to_knee_z, 4)
    robot_config["knee_to_ankle_z"] = round_to_sig_digits(knee_to_ankle_z, 4)
    robot_config["hip_to_ankle_pitch_z"] = round_to_sig_digits(hip_to_ankle_pitch_z, 4)
    robot_config["hip_to_ankle_roll_z"] = round_to_sig_digits(hip_to_ankle_roll_z, 4)
    robot_config["foot_to_com_y"] = round_to_sig_digits(foot_to_com_y, 4)

    # Clean up the temporary XML file
    if os.path.exists(robot_xml_path):
        os.remove(robot_xml_path)


def create_scene_xml(
    root: ET.Element,
    target_robot_dir: str,
    variant: str = "",
    important_contacts: List[str] = ["ankle_roll_link", "hand"],
):
    """Generates an XML scene file for a robot model based on the provided MJCF file path and configuration.

    Args:
        mjcf_path (str): The file path to the MJCF XML file of the robot model.
        is_fixed (bool): A flag indicating whether the robot is fixed in place. If True, adjusts camera positions and scene settings accordingly.

    Creates an XML scene file that includes the robot model, visual settings, and camera configurations. The scene is saved in the same directory as the input MJCF file with a modified filename.
    """
    robot_name = os.path.basename(target_robot_dir)
    # Create the root element
    mujoco = ET.Element("mujoco", attrib={"model": f"{robot_name}{variant}_scene"})
    # Include the robot model
    ET.SubElement(mujoco, "include", attrib={"file": f"{robot_name}{variant}.xml"})
    # Add statistic element
    center_z = -0.05 if "fixed" in variant else 0.25
    ET.SubElement(
        mujoco, "statistic", attrib={"center": f"0 0 {center_z}", "extent": "0.6"}
    )

    # Visual settings
    visual = ET.SubElement(mujoco, "visual")
    ET.SubElement(
        visual,
        "headlight",
        attrib={
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.3 0.3 0.3",
            "specular": "0 0 0",
        },
    )
    ET.SubElement(visual, "rgba", attrib={"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(
        visual,
        "global",
        attrib={
            "azimuth": "160",
            "elevation": "-20",
            "offwidth": "1280",
            "offheight": "720",
        },
    )

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        attrib={"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"},
    )

    camera_settings: Dict[str, Dict[str, List[float]]] = {
        "perspective": {"pos": [0.7, -0.7, 0.7], "xy_axes": [1, 1, 0, -1, 1, 3]},
        "side": {"pos": [0, -1, 0.6], "xy_axes": [1, 0, 0, 0, 1, 3]},
        "top": {"pos": [0, 0, 1], "xy_axes": [0, 1, 0, -1, 0, 0]},
        "front": {"pos": [1, 0, 0.6], "xy_axes": [0, 1, 0, -1, 0, 3]},
    }

    for camera, settings in camera_settings.items():
        pos_list = settings["pos"]
        if "fixed" in variant:
            pos_list = [pos_list[0], pos_list[1], pos_list[2] - 0.35]

        pos_str = " ".join(map(str, pos_list))
        xy_axes_str = " ".join(map(str, settings["xy_axes"]))

        ET.SubElement(
            worldbody,
            "camera",
            attrib={
                "name": camera,
                "pos": pos_str,
                "xyaxes": xy_axes_str,
                "mode": "trackcom",
            },
        )

    if "fixed" not in variant:
        # Worldbody settings
        ET.SubElement(
            worldbody,
            "geom",
            attrib={
                "name": "floor",
                "size": "0 0 0.05",
                "type": "plane",
                "material": "groundplane",
                "condim": "3",
            },
        )
        # Asset settings
        asset = ET.SubElement(mujoco, "asset")
        ET.SubElement(
            asset,
            "texture",
            attrib={
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.3 0.5 0.7",
                "rgb2": "0 0 0",
                "width": "512",
                "height": "3072",
            },
        )
        ET.SubElement(
            asset,
            "texture",
            attrib={
                "type": "2d",
                "name": "groundplane",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "0.2 0.3 0.4",
                "rgb2": "0.1 0.2 0.3",
                "markrgb": "0.8 0.8 0.8",
                "width": "300",
                "height": "300",
            },
        )
        ET.SubElement(
            asset,
            "material",
            attrib={
                "name": "groundplane",
                "texture": "groundplane",
                "texuniform": "true",
                "texrepeat": "5 5",
                "reflectance": "0.0",
            },
        )
        contact = ET.SubElement(mujoco, "contact")
        for body in root.findall(".//body"):
            geom_list = body.findall("./geom[@class='collision']")
            for geom in geom_list:
                geom_name = geom.attrib["name"]
                if "mjx" not in variant or any(
                    [k in geom_name for k in important_contacts]
                ):
                    ET.SubElement(
                        contact,
                        "pair",
                        attrib={"geom1": "floor", "geom2": geom_name},
                    )

    # Create a tree from the root element and write it to a file
    tree = ET.ElementTree(mujoco)
    pretty_write_xml(
        tree.getroot(),
        os.path.join(target_robot_dir, f"scene{variant}.xml"),
    )


def configure_motors(root: ET.Element):
    """Configures motor actuators by replacing position controllers with motor controllers."""
    default_elem = root.find("default")
    default_position = default_elem.find("position")
    default_elem.remove(default_position)
    ET.SubElement(default_elem, "motor", {"ctrlrange": "-10 10"})

    actuator_elem = root.find("actuator")
    motor_names = []
    motor_joints = []
    for position in actuator_elem.findall("position"):
        motor_names.append(position.get("name"))
        motor_joints.append(position.get("joint"))

    actuator_elem.clear()
    for name, joint in zip(motor_names, motor_joints):
        ET.SubElement(actuator_elem, "motor", {"name": name, "joint": joint})


def configure_fixed(root: ET.Element):
    """Configures robot for fixed-base simulation by removing free joint."""
    worldbody = root.find("worldbody")
    torso = worldbody.find("body")
    if torso is not None:
        # Move all children except freejoint and inertial
        for child in list(torso):
            if child.tag in ("freejoint", "inertial"):
                continue
            worldbody.append(child)
        worldbody.remove(torso)

    key = root.find(".//keyframe/key[@name='home']")
    if key is not None:
        qpos = key.attrib["qpos"].split()
        key.set("qpos", " ".join(qpos[7:]))


def configure_mjx(root: ET.Element, robot_config: dict):
    """Configures robot for MJX (GPU-accelerated MuJoCo) simulation."""

    def is_valid_pair(g1, g2):
        is_foot_foot = (
            robot_config["foot_name"] in g1 and robot_config["foot_name"] in g2
        )
        is_hand_hand = (
            robot_config["hand_name"] in g1 and robot_config["hand_name"] in g2
        )
        same_side = ("left" in g1 and "left" in g2) or ("right" in g1 and "right" in g2)
        is_foot_hand_same_side = (
            robot_config["foot_name"] in g1
            and robot_config["hand_name"] in g2
            and same_side
        ) or (
            robot_config["foot_name"] in g2
            and robot_config["hand_name"] in g1
            and same_side
        )
        return is_foot_foot or is_hand_hand or is_foot_hand_same_side

    contact = root.find("contact")
    for pair in list(contact.findall("pair")):
        geom1 = pair.get("geom1", "")
        geom2 = pair.get("geom2", "")
        if not is_valid_pair(geom1, geom2):
            contact.remove(pair)

    option = ET.SubElement(root, "option", {"iterations": "1", "ls_iterations": "4"})
    ET.SubElement(option, "flag", {"eulerdamp": "disable"})


def write_pos_xml(root: ET.Element, target_robot_dir: str):
    # Save to final path
    robot_name = os.path.basename(target_robot_dir)
    robot_xml_path = os.path.join(target_robot_dir, f"{robot_name}_pos.xml")
    pretty_write_xml(root, robot_xml_path)
    create_scene_xml(root, target_robot_dir, variant="_pos")


def write_xml(root: ET.Element, target_robot_dir: str):
    """Writes main robot XML file with motor configuration."""
    configure_motors(root)
    robot_name = os.path.basename(target_robot_dir)
    robot_xml_path = os.path.join(target_robot_dir, f"{robot_name}.xml")
    pretty_write_xml(root, robot_xml_path)
    create_scene_xml(root, target_robot_dir, variant="")


def write_mjx_xml(root: ET.Element, target_robot_dir: str, robot_config: dict):
    """Writes MJX-optimized robot XML file."""
    configure_motors(root)
    configure_mjx(root, robot_config)
    robot_name = os.path.basename(target_robot_dir)
    robot_xml_path = os.path.join(target_robot_dir, f"{robot_name}_mjx.xml")
    pretty_write_xml(root, robot_xml_path)
    create_scene_xml(root, target_robot_dir, variant="_mjx")


def write_pos_fixed_xml(root: ET.Element, target_robot_dir: str):
    configure_fixed(root)
    robot_name = os.path.basename(target_robot_dir)
    robot_xml_path = os.path.join(target_robot_dir, f"{robot_name}_pos_fixed.xml")
    pretty_write_xml(root, robot_xml_path)
    create_scene_xml(root, target_robot_dir, variant="_pos_fixed")


def write_fixed_xml(root: ET.Element, target_robot_dir: str):
    configure_motors(root)
    configure_fixed(root)
    robot_name = os.path.basename(target_robot_dir)
    robot_xml_path = os.path.join(target_robot_dir, f"{robot_name}_fixed.xml")
    pretty_write_xml(root, robot_xml_path)
    create_scene_xml(root, target_robot_dir, variant="_fixed")


def write_mjx_fixed_xml(root: ET.Element, target_robot_dir: str, robot_config: dict):
    configure_motors(root)
    configure_fixed(root)
    configure_mjx(root, robot_config)
    robot_name = os.path.basename(target_robot_dir)
    robot_xml_path = os.path.join(target_robot_dir, f"{robot_name}_mjx_fixed.xml")
    pretty_write_xml(root, robot_xml_path)
    create_scene_xml(root, target_robot_dir, variant="_mjx_fixed")


def assemble_xml(robot_name: str, torso_name: str, arm_name: str, leg_name: str):
    """Assembles a URDF file for a robot based on the provided configuration.

    This function constructs a complete URDF (Unified Robot Description Format) file by combining a base body URDF with optional arm and leg components specified in the configuration. It updates mesh file paths and ensures the correct structure for simulation.

    Raises:
        ValueError: If a source URDF for a specified link cannot be found.
    """
    # Parse the target URDF
    description_dir = os.path.join("toddlerbot", "descriptions")
    target_robot_dir = os.path.join(description_dir, robot_name)
    assembly_dir = os.path.join(description_dir, "assemblies")
    torso_xml_path = os.path.join(assembly_dir, torso_name, "robot.xml")
    torso_tree = ET.parse(torso_xml_path)
    torso_root = torso_tree.getroot()
    torso_root.set("model", robot_name)

    target_assets_dir = os.path.join(target_robot_dir, "assets")
    os.makedirs(target_assets_dir, exist_ok=True)

    global_config_path = os.path.join(description_dir, "default.yml")
    with open(global_config_path, "r") as f:
        global_config = yaml.safe_load(f)

    robot_config = {}
    motor_config_local = {}

    if "2xm" in robot_name:
        motor_config_local = {
            "left_hip_pitch": {"motor": "XM430-W350"},
            "right_hip_pitch": {"motor": "XM430-W350"},
            "left_hip_roll": {"motor": "XM430-W350"},
            "right_hip_roll": {"motor": "XM430-W350"},
            "left_shoulder_roll": {"motor": "2XC430"},
            "right_shoulder_roll": {"motor": "2XC430"},
            "left_shoulder_yaw_drive": {"motor": "2XC430"},
            "right_shoulder_yaw_drive": {"motor": "2XC430"},
            "left_elbow_roll": {"motor": "2XC430"},
            "right_elbow_roll": {"motor": "2XC430"},
            "left_elbow_yaw_drive": {"motor": "2XC430"},
            "right_elbow_yaw_drive": {"motor": "2XC430"},
            "left_wrist_pitch_drive": {"motor": "2XC430"},
            "right_wrist_pitch_drive": {"motor": "2XC430"},
            "left_wrist_roll": {"motor": "2XC430"},
            "right_wrist_roll": {"motor": "2XC430"},
        }

    # Check if the <mujoco> element already exists
    compiler = torso_root.find("compiler")
    compiler.set("meshdir", "assets")

    torso_defaults = torso_root.find("default")
    torso_defaults.clear()
    torso_defaults.set("class", "main")
    ET.SubElement(
        torso_defaults, "geom", {"condim": "1", "contype": "0", "conaffinity": "0"}
    )
    ET.SubElement(
        torso_defaults,
        "joint",
        {"damping": "0.01", "armature": "0.0", "frictionloss": "0.01"},
    )
    ET.SubElement(
        torso_defaults, "position", {"inheritrange": "1", "forcerange": "-10 10"}
    )
    vision_default = ET.SubElement(torso_defaults, "default", {"class": "visual"})
    ET.SubElement(vision_default, "geom", {"group": "2", "density": "0"})
    collision_default = ET.SubElement(torso_defaults, "default", {"class": "collision"})
    ET.SubElement(collision_default, "geom", {"group": "3"})

    for key, value in global_config["actuators"].items():
        if isinstance(value, dict) and "damping" in value:
            joint_default = ET.SubElement(torso_defaults, "default", {"class": key})
            ET.SubElement(
                joint_default,
                "joint",
                {
                    "damping": str(value["damping"]),
                    "armature": str(value["armature"]),
                    "frictionloss": str(value["frictionloss"]),
                },
            )

    torso_actuators = torso_root.find("actuator")
    for actuator in torso_actuators:
        del actuator.attrib["class"]
        del actuator.attrib["inheritrange"]
        kp_sim = (
            global_config["motors"][actuator.attrib["name"]]["kp"]
            / global_config["actuators"]["kp_ratio"]
        )
        actuator.set("kp", f"{kp_sim}")

    torso_equality = torso_root.find("equality")

    # Locate major sections
    torso_assets = torso_root.find("asset")
    redundant_assets = []
    for asset in torso_assets:
        if asset.tag == "mesh":
            file_path = asset.attrib["file"]
            if "leg" in os.path.basename(file_path) or "arm" in os.path.basename(
                file_path
            ):
                redundant_assets.append(asset)
            else:
                source_path = os.path.join(
                    assembly_dir, torso_name, "assets", file_path
                )
                shutil.copy2(source_path, target_assets_dir)
                asset.set("file", os.path.basename(file_path))
        else:
            name = asset.attrib.get("name")
            if "leg" in name or "arm" in name:
                redundant_assets.append(asset)

    # Remove redundant assets
    for asset in redundant_assets:
        torso_assets.remove(asset)

    torso_worldbody = torso_root.find("worldbody")
    for child_body in torso_worldbody:
        # Set the name and joint attributes for the body
        child_body.set("pos", list_to_string(global_config["kinematics"]["zero_pos"]))
        child_body.attrib["childclass"] = "main"
        stack = [child_body]
        while stack:
            nested_body = stack.pop()
            # Prefix body name

            child_to_add = []
            child_to_remove = []
            for child_idx, child in enumerate(nested_body):
                if child.tag == "joint" and "name" in child.attrib:
                    child_name = child.attrib["name"]
                    if child_name in global_config["motors"]:
                        motor_model = global_config["motors"][child_name]["motor"]
                        if (
                            child_name in motor_config_local
                            and "motor" in motor_config_local[child_name]
                        ):
                            motor_model = motor_config_local[child_name]["motor"]

                        child.set("class", motor_model)

                if child.tag == "geom" and child.attrib["class"] == "collision":
                    new_geom_list = update_collision(
                        child, global_config["collision"], target_assets_dir
                    )
                    geom_idx = child_idx
                    for new_geom in new_geom_list:
                        child_to_add.append((geom_idx, new_geom))
                        geom_idx += 1
                    child_to_remove.append(child)

                # Add nested body to stack
                if child.tag == "body":
                    stack.append(child)

            for child in child_to_remove:
                nested_body.remove(child)

            for child_idx, child in child_to_add:
                nested_body.insert(child_idx, child)

    # For each component (arm/leg), process both left and right
    limb_info = []
    if arm_name:
        limb_info.append(("arm", arm_name))
    if leg_name:
        limb_info.append(("leg", leg_name))

    material_names = set()
    for side in ["left", "right"]:
        for limb_type, limb_name in limb_info:
            if not limb_name:
                continue

            limb_id = f"{side}_{limb_type}_{limb_name}"
            limb_path = os.path.join(assembly_dir, limb_id, "robot.xml")
            limb_tree = ET.parse(limb_path)
            limb_root = limb_tree.getroot()

            # --- Append <actuator> ---
            limb_actuators = limb_root.find("actuator")
            if limb_actuators is not None:
                for actuator in limb_actuators:
                    del actuator.attrib["class"]
                    del actuator.attrib["inheritrange"]
                    actuator_name = f"{side}_{actuator.attrib['name']}"
                    kp_sim = (
                        global_config["motors"][actuator_name]["kp"]
                        / global_config["actuators"]["kp_ratio"]
                    )
                    actuator.set("name", actuator_name)
                    actuator.set("joint", f"{side}_{actuator.attrib['joint']}")
                    actuator.set("kp", f"{kp_sim}")
                    torso_actuators.append(actuator)

            # --- Append <equality> ---
            limb_equality = limb_root.find("equality")
            if limb_equality is not None:
                for eq in limb_equality:
                    if eq.tag == "joint":
                        eq.set("joint1", f"{side}_{eq.attrib['joint1']}")
                        eq.set("joint2", f"{side}_{eq.attrib['joint2']}")
                    torso_equality.append(eq)

            # --- Append <asset> ---
            limb_assets = limb_root.find("asset")
            if limb_assets is not None:
                for asset in limb_assets:
                    if asset.tag == "mesh":
                        source_path = os.path.join(
                            assembly_dir, limb_id, "assets", asset.attrib["file"]
                        )
                        target_name = f"{side}_{os.path.basename(asset.attrib['file'])}"
                        shutil.copy2(
                            source_path, os.path.join(target_assets_dir, target_name)
                        )
                        asset.set("file", target_name)
                    elif asset.tag == "material":
                        # Ensure unique material names
                        if asset.attrib["name"] in material_names:
                            continue

                        material_names.add(asset.attrib["name"])

                    torso_assets.append(asset)

            # --- Append <body> under worldbody ---
            limb_worldbody = limb_root.find("worldbody")
            if limb_worldbody is not None:
                for child_body in limb_worldbody:
                    # Set the name and joint attributes for the body
                    del child_body.attrib["childclass"]
                    stack = [child_body]
                    while stack:
                        nested_body = stack.pop()
                        if "name" in nested_body.attrib:
                            nested_body_name = nested_body.attrib["name"]
                            nested_body.set("name", f"{side}_{nested_body_name}")

                        child_to_add = []
                        child_to_remove = []
                        child_body_count = 0
                        for child_idx, child in enumerate(nested_body):
                            if child.tag == "joint" and "name" in child.attrib:
                                child_name = f"{side}_{child.attrib['name']}"
                                child.set("name", child_name)
                                if child_name in global_config["motors"]:
                                    motor_model = global_config["motors"][child_name][
                                        "motor"
                                    ]
                                    if (
                                        child_name in motor_config_local
                                        and "motor" in motor_config_local[child_name]
                                    ):
                                        motor_model = motor_config_local[child_name][
                                            "motor"
                                        ]

                                    child.set("class", motor_model)

                            if child.tag == "geom" and "mesh" in child.attrib:
                                child.set("mesh", f"{side}_{child.attrib['mesh']}")
                                if child.attrib["class"] == "collision":
                                    new_geom_list = update_collision(
                                        child,
                                        global_config["collision"],
                                        target_assets_dir,
                                    )
                                    geom_idx = child_idx
                                    for new_geom in new_geom_list:
                                        child_to_add.append((geom_idx, new_geom))
                                        geom_idx += 1
                                    child_to_remove.append(child)

                            # Add nested body to stack
                            if child.tag == "body":
                                stack.append(child)
                                child_body_count += 1

                        if (
                            "gripper" not in nested_body_name
                            and child_body_count == 0
                            and "drive" not in nested_body_name
                        ):
                            if limb_type == "arm":
                                robot_config["hand_name"] = nested_body_name
                            elif limb_type == "leg":
                                robot_config["foot_name"] = nested_body_name

                        if "gripper_link" in nested_body_name and limb_type == "arm":
                            robot_config["hand_name"] = nested_body_name

                        for child in child_to_remove:
                            nested_body.remove(child)

                        for child_idx, child in child_to_add:
                            nested_body.insert(child_idx, child)

                    suffix = "" if side == "left" else "_2"
                    target_name = f"{limb_type}{suffix}"
                    result = find_body_and_parent(torso_worldbody, target_name)
                    if result is None:
                        raise ValueError(f"Body with name '{target_name}' not found.")

                    parent, old_body = result

                    for old_child in old_body:
                        if old_child.tag == "joint":
                            old_joint = old_child
                            break

                    for old_child in old_body:
                        if old_child.tag == "geom":
                            old_geom_quat = string_to_list(old_child.attrib["quat"])
                            break

                    for child in child_body:
                        if child.tag == "geom":
                            new_geom_quat = string_to_list(child.attrib["quat"])
                            break

                    R_old = R.from_quat(
                        string_to_list(old_body.attrib["quat"]), scalar_first=True
                    )
                    R_old_geom = R.from_quat(old_geom_quat, scalar_first=True)
                    R_new_geom = R.from_quat(new_geom_quat, scalar_first=True)
                    R_new = R_old * R_old_geom * R_new_geom.inv()

                    old_joint_axis = string_to_list(old_joint.attrib["axis"])
                    new_joint_axis = (R_new.inv() * R_old).apply(old_joint_axis)
                    old_joint.set("axis", list_to_string(new_joint_axis))

                    child_body.set("pos", old_body.attrib["pos"])
                    child_body.set(
                        "quat", list_to_string(list(R_new.as_quat(scalar_first=True)))
                    )

                    # Remove the freejoint and insert the correct joint
                    for i, child in enumerate(child_body):
                        if child.tag == "freejoint":
                            child_body.remove(child)
                            child_body.insert(i, old_joint)
                            break

                    index = list(parent).index(old_body)
                    parent.remove(old_body)
                    parent.insert(index, child_body)

    sorted_meshes = []
    for elem in torso_assets:
        if elem.tag == "mesh":
            if (
                "collision" not in elem.attrib["file"]
                or "ankle_roll_link" in elem.attrib["file"]
            ):
                sorted_meshes.append(elem)
                if "right_ankle_roll_link" in elem.attrib["file"]:
                    shutil.copy2(
                        os.path.join(
                            target_assets_dir,
                            elem.attrib["file"].replace("right", "left"),
                        ),
                        os.path.join(target_assets_dir, elem.attrib["file"]),
                    )
            else:
                os.remove(os.path.join(target_assets_dir, elem.attrib["file"]))

    sorted_meshes = sorted(sorted_meshes, key=lambda el: el.attrib.get("file", ""))
    sorted_materials = sorted(
        [
            elem
            for elem in torso_assets
            if elem.tag == "material" and "collision" not in elem.attrib["name"]
        ],
        key=lambda el: el.attrib.get("name", ""),
    )

    torso_assets.clear()
    for elem in sorted_meshes + sorted_materials:
        torso_assets.append(elem)

    # Reorder torso actuators to match the ID list
    actuator_dict = {actuator.attrib["name"]: actuator for actuator in torso_actuators}
    torso_actuators.clear()
    home_pos = {}
    for motor_name, motor_config_global in global_config["motors"].items():
        if motor_name in actuator_dict:
            torso_actuators.append(actuator_dict[motor_name])
            home_pos[motor_name] = motor_config_global["home_pos"]

    if leg_name:
        add_waist_constraints(torso_root, global_config["kinematics"])

    if "gripper" in robot_config["hand_name"]:
        add_gripper_constraints(torso_root, global_config["kinematics"])

    update_joint_params(torso_root, global_config["joints"])

    add_sites(torso_root, global_config["kinematics"], robot_config)

    add_site_velocity_sensors(
        torso_root,
        [
            "torso_center",
            "left_hand_center",
            "right_hand_center",
            "left_foot_center",
            "right_foot_center",
        ],
    )

    torso_root = align_body_frames_to_global(torso_root)

    add_self_contacts(torso_root, global_config["contact"])

    if leg_name:
        add_offsets(torso_root, robot_config, target_robot_dir)

    add_keyframes(torso_root, home_pos, global_config["kinematics"], target_robot_dir)

    if leg_name:
        write_pos_xml(deepcopy(torso_root), target_robot_dir)
        write_xml(deepcopy(torso_root), target_robot_dir)
        write_mjx_xml(deepcopy(torso_root), target_robot_dir, robot_config)
        write_mjx_fixed_xml(deepcopy(torso_root), target_robot_dir, robot_config)

    write_pos_fixed_xml(deepcopy(torso_root), target_robot_dir)
    write_fixed_xml(deepcopy(torso_root), target_robot_dir)

    local_config = {"robot": robot_config}
    if len(motor_config_local) > 0:
        local_config["motors"] = motor_config_local

    with open(os.path.join(target_robot_dir, "robot.yml"), "w") as f:
        yaml.dump(
            local_config,
            f,
            indent=4,
            default_flow_style=False,
            sort_keys=False,
        )


def main():
    """Parses command-line arguments to configure and assemble a URDF (Unified Robot Description Format) file.

    This function sets up an argument parser to accept parameters for robot configuration, including the robot's name, body, arm, and leg components. It then calls the `assemble_xml` function with the parsed arguments to generate the URDF file.
    """
    parser = argparse.ArgumentParser(description="Assemble the xml.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot_2xc",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--torso-name",
        type=str,
        default="",
        help="The name of the torso.",
    )
    parser.add_argument(
        "--arm-name",
        type=str,
        default="",
        help="The name of the arm.",
    )
    parser.add_argument(
        "--leg-name",
        type=str,
        default="",
        help="The name of the leg.",
    )
    args = parser.parse_args()

    assemble_xml(args.robot, args.torso_name, args.arm_name, args.leg_name)


if __name__ == "__main__":
    main()
