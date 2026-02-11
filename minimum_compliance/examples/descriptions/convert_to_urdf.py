"""MuJoCo MJCF to URDF conversion utility.

Converts MuJoCo XML robot descriptions to URDF format for use with ROS and other
robotics frameworks. Handles geometry, joint, and inertial transformations between
the two formats.
"""

import argparse
import os
import warnings
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from toddlerbot.utils.io_utils import pretty_write_xml
from toddlerbot.utils.math_utils import round_to_sig_digits

# Adapted from https://github.com/Yasu31/mjcf_urdf_simple_converter

warnings.filterwarnings("ignore", message="Gimbal lock detected*")


def string_to_list(string: str) -> list:
    """Convert a comma-separated string to a list of strings."""
    return [float(item.strip()) for item in string.split(" ") if item.strip()]


def list_to_string(lst: list, digits: int = 6) -> str:
    """Convert a list of strings to a comma-separated string."""
    return " ".join(str(round_to_sig_digits(item, digits)) for item in lst)


def create_body(xml_root, name, inertial_pos, inertial_rpy, mass, ixx, iyy, izz):
    """Creates URDF link element with specified inertial properties.

    Args:
        xml_root: Parent XML element to attach link to
        name: Link name
        inertial_pos: Position of center of mass
        inertial_rpy: Orientation of inertial frame
        mass: Link mass
        ixx, iyy, izz: Principal moments of inertia

    Returns:
        Created link element
    """
    # create XML element for this body
    body = ET.SubElement(xml_root, "link", {"name": name})

    # add inertial element
    inertial = ET.SubElement(body, "inertial")
    ET.SubElement(
        inertial,
        "origin",
        {"xyz": list_to_string(inertial_pos), "rpy": list_to_string(inertial_rpy)},
    )
    ET.SubElement(inertial, "mass", {"value": str(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": str(ixx),
            "iyy": str(iyy),
            "izz": str(izz),
            "ixy": "0",
            "ixz": "0",
            "iyz": "0",
        },
    )
    return body


def create_dummy_body(xml_root, name):
    """Creates dummy URDF link with negligible mass for kinematic connections."""
    mass = 0.001
    mass_moi = mass * (0.001**2)  # mass moment of inertia
    return create_body(
        xml_root, name, np.zeros(3), np.zeros(3), mass, mass_moi, mass_moi, mass_moi
    )


def create_joint(
    xml_root, name, type, parent, child, pos, rpy, axis=None, jnt_range=None
):
    """Creates URDF joint element connecting parent and child links.

    Args:
        xml_root: Parent XML element
        name: Joint name
        type: Joint type (fixed, revolute, prismatic, floating)
        parent: Parent link name
        child: Child link name
        pos: Joint position
        rpy: Joint orientation
        axis: Joint axis for revolute/prismatic joints
        jnt_range: Joint limits [min, max]

    Returns:
        Created joint element
    """
    # create joint element connecting this to parent
    jnt_element = ET.SubElement(xml_root, "joint", {"type": type, "name": name})
    ET.SubElement(jnt_element, "parent", {"link": parent})
    ET.SubElement(jnt_element, "child", {"link": child})
    ET.SubElement(
        jnt_element, "origin", {"xyz": list_to_string(pos), "rpy": list_to_string(rpy)}
    )
    if type in ["revolute", "prismatic"]:
        ET.SubElement(jnt_element, "axis", {"xyz": list_to_string(axis)})
        ET.SubElement(
            jnt_element,
            "limit",
            {
                "lower": str(jnt_range[0]),
                "upper": str(jnt_range[1]),
                "effort": "100",
                "velocity": "100",
            },
        )
    elif type == "floating":
        ET.SubElement(
            jnt_element,
            "limit",
            {
                "lower": "-999999",
                "upper": "999999",
                "effort": "1000",
                "velocity": "1000",
            },
        )

    if name.endswith("_mirror"):
        # Add a special tag for mirrored joints
        ET.SubElement(
            jnt_element,
            "mimic",
            {"joint": name.replace("_mirror", ""), "multiplier": "1", "offset": "0"},
        )

    return jnt_element


def get_mesh_paths(root: ET.Element) -> tuple:
    """Extract mesh and texture information from MJCF/XML file.

    Returns:
        tuple: A tuple containing:
            - mesh_files (dict): Mapping of mesh names to their file paths
            - mesh_textures (dict): Mapping of mesh names to list of texture names they use
            - texture_files (dict): Mapping of texture names to their file paths
    """
    mesh_files = {}  # mesh_name -> file path
    materials = {}  # material_name -> dict with color info

    # First pass: collect all meshes and textures
    asset_elem = root.find(".//asset")
    if asset_elem is not None:
        # Get all meshes
        for mesh in asset_elem.findall("mesh"):
            mesh_name = mesh.attrib["file"].split(".")[0]  # Get the file name
            mesh_files[mesh_name] = mesh.attrib["file"]

        for material in asset_elem.findall("material"):
            materials[material.attrib["name"]] = material.attrib.get("rgba", "1 1 1 1")

    return mesh_files, materials


def get_geom_pose(root, geom_name):
    for geom in root.iter("geom"):
        if geom.attrib.get("mesh") == geom_name:
            pos_str = geom.attrib.get("pos", "0 0 0")
            quat_str = geom.attrib.get("quat", "1 0 0 0")
            pos = string_to_list(pos_str)
            quat = string_to_list(quat_str)
            return pos, quat

    raise ValueError(f"Geometry with name {geom_name} not found in MJCF file.")


def convert(robot_name: str, asset_file_prefix: str, fix_extra_joints: list = []):
    """Converts MuJoCo MJCF robot description to URDF format.

    Args:
        robot_name: Name of the robot to convert
        asset_file_prefix: Prefix path for asset files
        fix_extra_joints: List of joint keywords to fix (make non-actuated)
    """
    mjcf_file = os.path.join(
        "toddlerbot", "descriptions", robot_name, f"{robot_name}_pos.xml"
    )
    urdf_file = os.path.join(
        "toddlerbot", "descriptions", robot_name, f"{robot_name}.urdf"
    )
    mjcf_root = ET.parse(mjcf_file).getroot()
    urdf_root = ET.Element("robot", {"name": robot_name})

    model = mujoco.MjModel.from_xml_path(mjcf_file)

    mujoco_elem = ET.SubElement(urdf_root, "mujoco")
    ET.SubElement(
        mujoco_elem,
        "compiler",
        {"strippath": "false", "balanceinertia": "true", "discardvisual": "false"},
    )

    # Get mesh and texture paths
    mesh_files, materials = get_mesh_paths(mjcf_root)

    for id in range(model.nbody):
        child_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id)
        parent_id = model.body_parentid[id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)

        # URDFs assume that the link origin is at the joint position, while in MJCF they can have user-defined values
        # this requires some conversion for the visual, inertial, and joint elements...
        # this is done by creating a dummy body with negligible mass and inertia at the joint position.
        parentbody2childbody_pos = model.body_pos[id]
        parentbody2childbody_quat = model.body_quat[id]  # [w, x, y, z]
        # change to [x, y, z, w]
        parentbody2childbody_quat = [
            parentbody2childbody_quat[1],
            parentbody2childbody_quat[2],
            parentbody2childbody_quat[3],
            parentbody2childbody_quat[0],
        ]
        parentbody2childbody_rot = R.from_quat(parentbody2childbody_quat).as_matrix()
        parentbody2childbody_rpy = R.from_matrix(parentbody2childbody_rot).as_euler(
            "xyz"
        )

        # read inertial info
        mass = model.body_mass[id]
        inertia = model.body_inertia[id]
        childbody2childinertia_pos = model.body_ipos[id]
        childbody2childinertia_quat = model.body_iquat[id]  # [w, x, y, z]
        # change to [x, y, z, w]
        childbody2childinertia_quat = [
            childbody2childinertia_quat[1],
            childbody2childinertia_quat[2],
            childbody2childinertia_quat[3],
            childbody2childinertia_quat[0],
        ]
        childbody2childinertia_rot = R.from_quat(
            childbody2childinertia_quat
        ).as_matrix()
        childbody2childinertia_rpy = R.from_matrix(childbody2childinertia_rot).as_euler(
            "xyz"
        )

        # create child body
        body_element = create_body(
            urdf_root,
            child_name,
            childbody2childinertia_pos,
            childbody2childinertia_rpy,
            mass,
            inertia[0],
            inertia[1],
            inertia[2],
        )

        # read geom info and add it child body
        visual_dict_list = []
        collision_dict_list = []
        geomnum = model.body_geomnum[id]
        for geomnum_i in range(geomnum):
            geomid = model.body_geomadr[id] + geomnum_i
            geom_dataid = model.geom_dataid[geomid]  # id of geom's mesh
            geom_type = model.geom_type[geomid]
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                # Due to the way MuJoCo handles meshes, we need to get the mesh pose
                # from the MJCF file
                mesh_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_MESH, geom_dataid
                )
                geom_pos, geom_quat = get_geom_pose(mjcf_root, mesh_name)
            else:
                geom_pos = model.geom_pos[geomid]
                geom_quat = model.geom_quat[geomid]  # [w, x, y, z]

            geom_quat = [
                geom_quat[1],  # x
                geom_quat[2],  # y
                geom_quat[3],  # z
                geom_quat[0],  # w
            ]
            geom_rpy = R.from_quat(geom_quat).as_euler("xyz")
            geom_size = model.geom_size[geomid]
            geom_group = model.geom_group[geomid]
            geom_matid = model.geom_matid[geomid]

            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                geom_dict = {
                    "type": "sphere",
                    "radius": str(geom_size[0]),  # First element is radius for sphere
                }

            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # URDF doesn't support capsules directly, so we'll use a cylinder
                geom_dict = {
                    "type": "cylinder",
                    "radius": str(geom_size[0]),  # First element is radius
                    "length": str(
                        geom_size[1] * 2
                    ),  # Second element is half-length in MuJoCo
                }

            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom_dict = {
                    "type": "cylinder",
                    "radius": str(geom_size[0]),  # First element is radius
                    "length": str(
                        geom_size[1] * 2
                    ),  # Second element is half-length in MuJoCo
                }

            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                geom_dict = {
                    "type": "box",
                    "size": list_to_string(  # MuJoCo uses half-sizes
                        [
                            geom_size[0] * 2,
                            geom_size[1] * 2,
                            geom_size[2] * 2,
                        ]
                    ),
                }

            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                rel_mesh_path = os.path.join(asset_file_prefix, mesh_files[mesh_name])
                abs_mesh_path = os.path.join(os.path.dirname(mjcf_file), rel_mesh_path)

                if os.path.exists(abs_mesh_path):
                    geom_dict = {
                        "type": "mesh",
                        "filename": rel_mesh_path,
                        "scale": "1.0 1.0 1.0",
                    }
                else:
                    raise FileNotFoundError(
                        f"Mesh file {abs_mesh_path} not found for mesh {mesh_name}."
                    )

            else:
                raise ValueError(
                    f"Unsupported geometry type {geom_type} for link {child_name}. "
                    "Only sphere, capsule, cylinder, box, and mesh are supported."
                )

            if geom_group == 2:
                # This is a visual element
                visual_name = f"{child_name}_visual"
                visual_dict = {
                    "name": visual_name,
                    "origin_xyz": geom_pos,
                    "origin_rpy": geom_rpy,
                    "geometry": geom_dict,
                }
                mat_adr = model.name_matadr[geom_matid]
                if geom_matid + 1 < model.nmat:
                    mat_end = model.name_matadr[geom_matid + 1]
                    mat_name = (
                        model.names[mat_adr:mat_end].decode("utf-8").replace("\x00", "")
                    )
                else:
                    mat_name = model.names[mat_adr:].decode("utf-8").split("\x00")[0]

                visual_dict["material"] = mat_name
                visual_dict_list.append(visual_dict)

            elif geom_group == 3:
                # This is a collision element
                collision_name = f"{child_name}_collision"
                collision_dict = {
                    "name": collision_name,
                    "origin_xyz": geom_pos,
                    "origin_rpy": geom_rpy,
                    "geometry": geom_dict,
                }
                collision_dict_list.append(collision_dict)

        if len(visual_dict_list) > 1:
            for i, visual_dict in enumerate(visual_dict_list):
                visual_dict["name"] = f"{visual_dict['name']}_{i}"

        if len(collision_dict_list) > 1:
            for i, collision_dict in enumerate(collision_dict_list):
                collision_dict["name"] = f"{collision_dict['name']}_{i}"

        for type, geom_dict in zip(
            ["visual"] * len(visual_dict_list)
            + ["collision"] * len(collision_dict_list),
            visual_dict_list + collision_dict_list,
        ):
            geom_el = ET.SubElement(body_element, type, {"name": geom_dict["name"]})
            ET.SubElement(
                geom_el,
                "origin",
                {
                    "xyz": list_to_string(geom_dict["origin_xyz"]),
                    "rpy": list_to_string(geom_dict["origin_rpy"]),
                },
            )
            mesh_el = ET.SubElement(geom_el, "geometry")
            if geom_dict["geometry"]["type"] == "mesh":
                ET.SubElement(
                    mesh_el,
                    "mesh",
                    {
                        "filename": geom_dict["geometry"]["filename"],
                        "scale": "1.0 1.0 1.0",
                    },
                )
            else:
                ET.SubElement(
                    mesh_el,
                    geom_dict["geometry"]["type"],
                    geom_dict["geometry"],
                )

            if "material" in geom_dict:
                ET.SubElement(geom_el, "material", {"name": geom_dict["material"]})

        jntnum = model.body_jntnum[id]

        if child_name == "world":
            # there is no joint connecting the world to anything, since it is the root
            assert parent_name == "world"
            assert jntnum == 0
            continue  # skip adding joint element or parent body

        if jntnum == 0:
            # No joints, create a fixed joint directly to parent
            jnt_name = f"{parent_name}2{child_name}_fixed"
            parentbody2jnt_pos = parentbody2childbody_pos
            parentbody2jnt_rpy = parentbody2childbody_rpy
            create_joint(
                urdf_root,
                jnt_name,
                "fixed",
                parent_name,
                child_name,
                parentbody2jnt_pos,
                parentbody2jnt_rpy,
            )
        elif jntnum == 1:
            jntid = model.body_jntadr[id]
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jntid)
            if jnt_name is None:
                # Generate a random name for the joint
                jnt_name = f"joint_{jntid}"
                print(
                    f"WARNING: joint name for {jntid} is None (could happen for ball joints with >1DoF), using automatically generated name {jnt_name}"
                )

            jnt_body_name = f"{jnt_name}_body"
            # Create dummy body for this joint
            create_dummy_body(urdf_root, jnt_body_name)

            fix = False
            if jnt_name and len(fix_extra_joints) > 0:
                jnt_words = jnt_name.split("_")
                for key in fix_extra_joints:
                    if key in jnt_words:
                        print(f"Fix joint {jnt_name} from {child_name}.")
                        fix = True
                        break

            if fix:
                jnt_type = "fixed"
            elif model.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_HINGE:
                jnt_type = "revolute"
            elif model.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_SLIDE:
                jnt_type = "prismatic"
            elif model.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_FREE:
                jnt_type = "floating"
            else:
                jnt_type = "fixed"

            if jnt_type in ["revolute", "prismatic"]:
                # Revolute joint
                jnt_range = model.jnt_range[jntid]  # [min, max]
                jnt_axis_childbody = model.jnt_axis[jntid]  # [x, y, z]
                childbody2jnt_pos = model.jnt_pos[jntid]  # [x, y, z]

                # First joint connects to original parent
                parentbody2jnt_pos = (
                    parentbody2childbody_pos
                    + parentbody2childbody_rot @ childbody2jnt_pos
                )
                parentbody2jnt_rpy = parentbody2childbody_rpy
                parentbody2jnt_axis = jnt_axis_childbody  # In child body frame

            else:
                childbody2jnt_pos = model.jnt_pos[jntid]
                parentbody2jnt_pos = parentbody2childbody_pos
                parentbody2jnt_rpy = parentbody2childbody_rpy

                parentbody2jnt_axis = None
                jnt_range = None

            # Connect current parent to this joint body
            create_joint(
                urdf_root,
                jnt_name,
                jnt_type,
                parent_name,
                jnt_body_name,
                parentbody2jnt_pos,
                parentbody2jnt_rpy,
                parentbody2jnt_axis,
                jnt_range,
            )

            # Connect last dummy body to child body with fixed joint
            # "bring back" the body coordinates to the child body frame
            jnt2childbody_pos = -childbody2jnt_pos if jntnum > 0 else np.zeros(3)
            jnt2childbody_rpy = np.zeros(3)
            create_joint(
                urdf_root,
                f"{jnt_name}_offset",
                "fixed",
                jnt_body_name,
                child_name,
                jnt2childbody_pos,
                jnt2childbody_rpy,
            )
        else:
            raise ValueError(
                f"Unsupported number of joints {jntnum} for link {child_name}. "
                "Only 0 or 1 joint is supported."
            )

    for mat_name, mat_rgba in materials.items():
        material_el = ET.SubElement(urdf_root, "material", {"name": mat_name})
        ET.SubElement(material_el, "color", {"rgba": mat_rgba})

    pretty_write_xml(urdf_root, urdf_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="")
    args = parser.parse_args()

    convert(
        args.robot,
        "assets",
        fix_extra_joints=["drive", "act", "front", "back", "rack"],
    )
