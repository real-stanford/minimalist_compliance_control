"""Complete OnShape-to-robot conversion pipeline.

Orchestrates the full robot description generation workflow from OnShape CAD to
final XML and URDF files. Provides interactive prompts for each conversion step
and supports multiple robot configurations.
"""

import argparse
import subprocess
import sys

ASSEMBLY_DOC_MAP = {
    "2xc_430_palm": "565bc33af293a651f66e88d2",
    "2xc_430_gripper": "565bc33af293a651f66e88d2",
    "2xm_430_palm": "565bc33af293a651f66e88d2",
    "2xm_430_gripper": "565bc33af293a651f66e88d2",
    "teleop_leader": "4d8474461f9a66aa40a5d850",
    "left_leg_2xc_430": "3084b13ad43394bd46cc00cf",
    "right_leg_2xc_430": "3084b13ad43394bd46cc00cf",
    "left_leg_2xm_430": "3084b13ad43394bd46cc00cf",
    "right_leg_2xm_430": "3084b13ad43394bd46cc00cf",
    "left_arm_palm": "9d5acd05cb0e8be60d56ce56",
    "right_arm_palm": "9d5acd05cb0e8be60d56ce56",
    "left_arm_gripper": "9d5acd05cb0e8be60d56ce56",
    "right_arm_gripper": "9d5acd05cb0e8be60d56ce56",
    "left_arm_leader": "9d5acd05cb0e8be60d56ce56",
    "right_arm_leader": "9d5acd05cb0e8be60d56ce56",
    "sysID_XC330": "1fb5d9a88ac086a053c4340b",
    "sysID_XC430": "1fb5d9a88ac086a053c4340b",
    "sysID_2XC430": "1fb5d9a88ac086a053c4340b",
    "sysID_2XL430": "1fb5d9a88ac086a053c4340b",
    "sysID_XM430": "1fb5d9a88ac086a053c4340b",
}

ROBOT_CONFIGS = {
    "toddlerbot_2xc": {"body": "2xc_430_palm", "arm": "palm", "leg": "2xc_430"},
    "toddlerbot_2xc_gripper": {
        "body": "2xc_430_gripper",
        "arm": "gripper",
        "leg": "2xc_430",
    },
    "toddlerbot_2xm": {"body": "2xm_430_palm", "arm": "palm", "leg": "2xm_430"},
    "toddlerbot_2xm_gripper": {
        "body": "2xm_430_gripper",
        "arm": "gripper",
        "leg": "2xm_430",
    },
    "teleop_leader": {"body": "teleop_leader", "arm": "leader"},
    "sysID_XC330": {"body": "sysID_XC330"},
    "sysID_XC430": {"body": "sysID_XC430"},
    "sysID_2XC430": {"body": "sysID_2XC430"},
    "sysID_2XL430": {"body": "sysID_2XL430"},
    "sysID_XM430": {"body": "sysID_XM430"},
}


def prompt_yes_no(prompt):
    """Prompts user for yes/no input and returns boolean result."""
    return input(f"{prompt} (y/n) > ").strip().lower() == "y"


def main():
    """Main entry point for OnShape-to-robot conversion pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert OnShape assemblies to URDF and MJCF"
    )
    parser.add_argument(
        "--robot",
        help="Robot name (e.g., toddlerbot_2xc)",
        default="",
    )
    parser.add_argument(
        "--assembly",
        nargs="*",
        help="Optional list of specific assemblies",
        default=None,
    )
    args = parser.parse_args()

    robot = args.robot
    assemblies = args.assembly

    if robot not in ROBOT_CONFIGS:
        print(f"❌ Unknown robot name: {robot}")
        sys.exit(1)

    config = ROBOT_CONFIGS[robot]
    body = config["body"]
    arm = config.get("arm", None)
    leg = config.get("leg", None)

    if not assemblies:
        assemblies = [body]
        if leg:
            assemblies += [f"left_leg_{leg}", f"right_leg_{leg}"]
        if arm:
            assemblies += [f"left_arm_{arm}", f"right_arm_{arm}"]

    doc_ids = []
    for name in assemblies:
        if name not in ASSEMBLY_DOC_MAP:
            print(f"❌ Unknown assembly: {name}")
            sys.exit(1)
        doc_ids.append(ASSEMBLY_DOC_MAP[name])

    repo = "toddlerbot"
    if prompt_yes_no("Do you want to export XML files from OnShape?"):
        subprocess.run(
            [
                "bash",
                "-c",
                f"source ~/.bashrc && python {repo}/descriptions/get_xml.py "
                f"--doc-id-list {' '.join(doc_ids)} "
                f"--assembly-list {' '.join(assemblies)}",
            ]
        )
        print("\nExport completed.\n")
    else:
        print("\nExport skipped.\n")

    if prompt_yes_no("Do you want to assemble the XML files?"):
        cmd = [
            "python",
            f"{repo}/descriptions/assemble_xml.py",
            "--robot",
            robot,
            "--torso-name",
            body,
        ]
        if arm:
            cmd += ["--arm-name", arm]
        if leg:
            cmd += ["--leg-name", leg]
        subprocess.run(cmd)
        print("\nAssembly completed.\n")
    else:
        print("\nAssembly skipped.\n")

    if prompt_yes_no("Do you want to visualize the XML?"):
        xml_path = f"{repo}/descriptions/{robot}/scene_pos_fixed.xml"
        subprocess.run(["python", "-m", "mujoco.viewer", "--mjcf", xml_path])

    if prompt_yes_no("Do you want to convert the XML to URDF?"):
        subprocess.run(
            ["python", f"{repo}/descriptions/convert_to_urdf.py", "--robot", robot]
        )
        print("\nConversion to URDF completed.\n")

    if prompt_yes_no("Do you want to visualize the URDF?"):
        urdf_path = f"{repo}/descriptions/{robot}/{robot}.urdf"
        subprocess.run(["python", "-m", "mujoco.viewer", "--mjcf", urdf_path])


if __name__ == "__main__":
    main()
