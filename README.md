# minimalist_compliance_control

**[Project Page](https://minimalist-compliance-control.github.io/)** |
**[Paper](https://arxiv.org/abs/2603.00913)** |
**[Tweet/X](https://x.com/HaochenShi74/status/2028726677749330388?s=20)**

A lightweight package for MuJoCo-based compliance control and wrench estimation.

The method estimates external wrenches from motor current/voltage and Jacobians, requires no force sensors or learning, and is plug-and-play with VLM, imitation, and model-based policies across tasks like wiping, drawing, scooping, and in-hand manipulation.

![Teaser](assets/teaser_release.gif)

## Overview

`minimalist_compliance_control` provides:

- Wrench simulation and Jacobian utilities,
- Online wrench estimation,
- Compliance reference integration,
- Unified policy/controller orchestration.

## Installation

```bash
conda create -n mcc python=3.10
conda activate mcc
pip install -e .
```

For policy stacks (model-based / diffusion-policy / VLM):

```bash
pip install -e ".[policy]"
```

To include the Dynamixel C++ backend:

```bash
pip install -e ".[policy]" --config-settings=cmake.define.BUILD_DYNAMIXEL=ON
```

## Policy Scripts

`policy/run_policy.py` is the main entrypoint for all policy variants.
For best performance, make sure your policy loop can sustain `50 Hz`.
This repository is designed primarily for real-world execution, and
real-world performance is often better than in simulation.

### Compliance

- `policy/compliance.py`: base compliance policy implementation (`--policy compliance`).
- Examples:
  ```bash
  mcc-run-policy --policy compliance --robot leap --sim mujoco --vis view
  mcc-run-policy --policy compliance_model_based --robot toddlerbot --sim mujoco --vis view
  mcc-run-policy --policy compliance_dp --robot toddlerbot --sim real --ckpt /path/to/ckpt.pth
  mcc-run-policy --policy compliance_vlm --robot toddlerbot --sim real --object "star" --site-names "right_hand_center"
  mcc-run-policy --policy compliance --robot toddlerbot --sim real --vis none
  mcc-run-policy --policy compliance --robot arx --sim real --vis none
  mcc-run-policy --policy compliance --robot g1 --sim real --vis none --ip en0
  ```
  - **Important:** For best performance, make sure your policy loop can sustain
    `50 Hz`.
  - Equivalent direct script invocation:
  ```bash
  python policy/run_policy.py --policy compliance --robot leap --sim mujoco --vis view
  ```

## Real-World Setup

For ARX, install the external ARX5 SDK: https://github.com/real-stanford/arx5-sdk.
If you cannot find the CAN interface, try `sudo ip link set up can0 type can bitrate 1000000`.
Example ARX real-run environment:
```bash
export ARX5_SDK_PATH=/home/haochen/arx5-sdk/python
export AMENT_PREFIX_PATH=/home/haochen/miniforge3/envs/arx-py310
export ARX5_INTERFACE=can0   # optional but recommended
mcc-run-policy --policy compliance --robot arx --sim real
```
For ToddlerBot, LEAP, and Unitree G1, real-world support is included in this repository.

### Compliance With VLM (Real World Only)

- `policy/compliance_vlm.py`
  - VLM-guided compliance implementation (`--policy compliance_vlm`).
- `policy/run_affordance_prediction.py`
  - Offline affordance prediction + EE pose planning from stereo images in `assets/`.
- Examples:
  ```bash
  python policy/run_affordance_prediction.py --robot toddlerbot --task wipe
  python policy/run_affordance_prediction.py --robot leap --task draw --site rf_tip if_tip --object "star"
  mcc-run-policy --policy compliance_vlm --robot toddlerbot --sim real
  ```

### Compliance With DP (Real World Only)

- `policy/compliance_dp.py`
  - Diffusion-policy compliance implementation (`--policy compliance_dp`).
- Example:
  ```bash
  mcc-run-policy --policy compliance_dp --robot toddlerbot --sim real --ckpt /path/to/ckpt.pth
  ```

### Compliance With Model-Based Planning

- `policy/compliance_model_based.py`
  - Model-based policy selector wrapper (`--policy compliance_model_based`).
- `policy/compliance_model_based_leap.py`
  - LEAP-specific model-based implementation.
- `policy/compliance_model_based_toddlerbot.py`
  - Toddlerbot-specific model-based implementation.
- Example:
  ```bash
  mcc-run-policy --policy compliance_model_based --robot toddlerbot --sim mujoco --vis view
  ```

## Data and Checkpoints
- Download shared assets from [Google Drive](https://drive.google.com/drive/folders/1nikob3DbPNiTi6TTQ6c3C2N-hcn91RBn?usp=drive_link).
- Extract `robologger.zip` into `datasets/`, then train with:
    ```bash
    python diffusion_policy/train.py --dataset-paths datasets/robologger/compliance_follower_20260110_140133
    ```
- Extract `toddlerbot_2xm_dp_20260113_141759.zip` into `results/` for the diffusion policy checkpoint.
- Place the foundation stereo engine at `ckpts/foundation_stereo_vitl_480x640_20.engine`.
- Set API keys for affordance/compliance providers: `GOOGLE_API_KEY` and `OPENAI_API_KEY` (if using the OpenAI provider).

## Related Folders

- `policy/`: policy implementations and policy utilities.
- `sim/`: simulation adapters (`base_sim.py`, `sim.py`) used by `run_policy.py`.
- `hybrid_servo/`: model-based algorithms and utilities.
- `diffusion_policy/`: diffusion model components.
- `vlm/`: VLM affordance/depth/servers.
- `real_world/`: hardware adapters (`real_world_dynamixel.py`,
  `real_world_arx.py`, `real_world_g1.py`) and IMU/camera interfaces.

## Related Projects

- [ToddlerBot](https://toddlerbot.github.io/)
- [Robot Trains Robot](https://robot-trains-robot.github.io/)
- [Locomotion Beyond Feet](https://locomotion-beyond-feet.github.io/)

## Citation
```bibtex
@misc{shi2026minimalist,
  title = {Minimalist {{Compliance Control}}},
  author = {Shi, Haochen and Hu, Songbo and Hou, Yifan and Wang, Weizhuo and Liu, Karen and Song, Shuran},
  year = 2026,
  month = mar,
  number = {arXiv:2603.00913},
  eprint = {2603.00913},
  primaryclass = {cs},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2603.00913},
  urldate = {2026-03-03},
  archiveprefix = {arXiv},
  keywords = {Computer Science - Robotics}
}
```
