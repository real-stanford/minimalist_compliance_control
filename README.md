# minimalist_compliance_control

A lightweight package for MuJoCo-based compliance control and wrench estimation.

## Overview

`minimalist_compliance_control` provides a minimal, reusable control stack with:

- wrench simulation and Jacobian utilities,
- online wrench estimation,
- compliance reference integration,
- a unified controller interface.

## Installation

```bash
conda create -n mcc python=3.10
conda activate mcc
pip install -e .
```

For example stacks (model-based / diffusion-policy / VLM):

```bash
pip install -e ".[examples]"
```

To include the Dynamixel C++ backend as well:

```bash
pip install -e ".[examples]" --config-settings=cmake.define.BUILD_DYNAMIXEL=ON
```

Installed CLI entrypoints:

```bash
mcc-run-policy --policy compliance --robot leap --sim mujoco --vis
mcc-run-policy --policy compliance_model_based --robot toddlerbot
mcc-run-policy --policy compliance_model_based --robot leap -- --scene-xml descriptions/leap_hand/scene_object_fixed.xml
mcc-run-policy --policy compliance_dp --robot toddlerbot -- --help
mcc-run-policy --policy compliance_vlm --robot toddlerbot -- --help
```

## Core Modules

- `minimalist_compliance_control/wrench_sim.py`
- `minimalist_compliance_control/wrench_estimation.py`
- `minimalist_compliance_control/controller.py`
- `minimalist_compliance_control/compliance_ref.py`
- `minimalist_compliance_control/ik_solvers.py`

## Examples

Example usage is documented under:

- `hybrid_servo/`
- `diffusion_policy/`
- `vlm/`

Hardware support modules are in `real_world/` (Dynamixel, IMU, camera).
