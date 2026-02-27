# minimalist_compliance_control

A lightweight package for MuJoCo-based compliance control and wrench estimation.

## Project Links

- Project page: https://minimalist-compliance-control.github.io/
- Paper: coming soon (project-page `Paper` button currently has no public URL)
- Tweet/X: coming soon (project-page `Tweet` button currently has no public URL)

## Overview

`minimalist_compliance_control` provides:

- wrench simulation and Jacobian utilities,
- online wrench estimation,
- compliance reference integration,
- unified policy/controller orchestration.

From the project page: the method estimates external wrenches from motor
current/voltage and Jacobians, requires no force sensors or learning, and is
plug-and-play with VLM, imitation, and model-based policies across tasks like
wiping, drawing, scooping, and in-hand manipulation.

## Teaser Video

<video src="assets/teaser_release.mp4" controls muted loop playsinline width="100%"></video>

Direct file: [assets/teaser_release.mp4](assets/teaser_release.mp4)

## Citation

Until the paper URL is published on the project page, you can cite the project
page entry:

```bibtex
@misc{shi2026minimalist_compliance_control,
  title        = {Minimalist Compliance Control},
  author       = {Shi, Haochen and Hu, Songbo and Hou, Yifan and Wang, Weizhuo and Liu, C. Karen and Song, Shuran},
  year         = {2026},
  howpublished = {\url{https://minimalist-compliance-control.github.io/}},
  note         = {Project page}
}
```

## Related Projects

- ToddlerBot: https://toddlerbot.github.io/
- Robot Trains Robot: https://robot-trains-robot.github.io/
- Locomotion Beyond Feet: https://locomotion-beyond-feet.github.io/

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

Complete coverage of files under `policy/`:

- `policy/run_policy.py`
  - Main runner for policy + sim/hardware backends.
  - Run with CLI entrypoint:
  ```bash
  mcc-run-policy --policy compliance --robot leap --sim mujoco --vis view
  mcc-run-policy --policy compliance_model_based --robot toddlerbot --sim mujoco --vis view
  mcc-run-policy --policy compliance_dp --robot toddlerbot --sim mujoco --ckpt /path/to/ckpt.pth
  mcc-run-policy --policy compliance_vlm --robot toddlerbot --sim mujoco --object "star" --site-names "right_hand_center"
  mcc-run-policy --policy compliance --robot toddlerbot --sim real --vis none
  mcc-run-policy --policy compliance --robot arx --sim real --vis none
  mcc-run-policy --policy compliance --robot g1 --sim real --vis none --ip en0
  ```
  - Equivalent direct script invocation:
  ```bash
  python policy/run_policy.py --policy compliance --robot leap --sim mujoco --vis view
  ```

- `policy/run_affordance_prediction.py`
  - Offline affordance prediction + EE pose planning from stereo images in `assets/`.
  - Example:
  ```bash
  python policy/run_affordance_prediction.py --robot toddlerbot --task wipe --provider gemini --model gemini-2.5-pro
  python policy/run_affordance_prediction.py --robot leap --task draw --site rf_tip if_tip --object "star"
  ```

- `policy/plot_log_data.py`
  - Plot `log_data.lz4` produced by `run_policy.py`.
  - Example:
  ```bash
  python policy/plot_log_data.py --log results/<run_dir>/log_data.lz4
  ```

- `policy/compliance.py`
  - Base compliance policy implementation.
  - Loaded by `run_policy.py` when `--policy compliance`.

- `policy/compliance_model_based.py`
  - Model-based policy selector wrapper.
  - Loaded by `run_policy.py` when `--policy compliance_model_based`.

- `policy/compliance_model_based_leap.py`
  - LEAP-specific model-based implementation.
  - Used by `policy/compliance_model_based.py` for `--robot leap`.

- `policy/compliance_model_based_toddlerbot.py`
  - Toddlerbot-specific model-based implementation.
  - Used by `policy/compliance_model_based.py` for `--robot toddlerbot`.

- `policy/compliance_dp.py`
  - Diffusion-policy compliance implementation.
  - Loaded by `run_policy.py` when `--policy compliance_dp`.

- `policy/compliance_vlm.py`
  - VLM-guided compliance implementation.
  - Loaded by `run_policy.py` when `--policy compliance_vlm`.

- `policy/__init__.py`
  - Package marker.

## Assets And Checkpoints

- Diffusion policy checkpoint: place under `results/`.
- Foundation stereo engine path:
  - `ckpts/foundation_stereo_vitl_480x640_20.engine`
- API keys for affordance/compliance providers:
  - `GOOGLE_API_KEY`
  - `OPENAI_API_KEY` (if using OpenAI provider)

## Core Modules

- `minimalist_compliance_control/wrench_sim.py`
- `minimalist_compliance_control/wrench_estimation.py`
- `minimalist_compliance_control/controller.py`
- `minimalist_compliance_control/compliance_ref.py`
- `minimalist_compliance_control/ik_solvers.py`

## Related Folders

- `policy/`: policy implementations and policy utilities.
- `sim/`: simulation adapters (`base_sim.py`, `sim.py`) used by `run_policy.py`.
- `hybrid_servo/`: model-based algorithms and utilities.
- `diffusion_policy/`: diffusion model components.
- `vlm/`: VLM affordance/depth/servers.
- `real_world/`: hardware adapters (`real_world_dynamixel.py`,
  `real_world_arx.py`, `real_world_g1.py`) and IMU/camera interfaces.
