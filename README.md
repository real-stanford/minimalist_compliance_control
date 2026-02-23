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

## Diffusion Policy Checkpoint

Download and unzip the diffusion policy checkpoint under `results/`:

```bash
cd results
gdown --fuzzy https://drive.google.com/file/d/1c-NxnbCkwnZ9I5qnSQABOluMff5IX23k/view?usp=drive_link && unzip "$(ls -t *.zip | head -n 1)"
```

## Foundation Stereo Engine

The foundation stereo server requires the TensorRT engine file at:

- `ckpts/foundation_stereo_vitl_480x640_20.engine`

Download it with:

```bash
mkdir -p ckpts
cd ckpts
gdown --fuzzy "https://drive.google.com/file/d/1fHppa6f15CLT8LnDXHmoAMfA2hoUsDrC/view?usp=drive_link"
```

## SAM3 Setup

For SAM3 installation, follow the official instructions from:

- https://github.com/facebookresearch/sam3

## API Key

When using the Gemini-backed affordance/compliance pipeline, set:

- `GOOGLE_API_KEY`

Example:

```bash
export GOOGLE_API_KEY="your_api_key_here"
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
