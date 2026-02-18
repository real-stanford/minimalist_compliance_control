# Compliance DP Example

## Purpose

This example runs a standalone diffusion-based compliance policy and outputs
`pose_command` and `command_matrix` compatible with the package compliance layout.

## Usage

Install required dependencies:

```bash
pip install torch torchvision diffusers opencv-python joblib
```

Replay mode:

```bash
python examples/run_policy.py --policy compliance_dp --robot toddlerbot -- \
  --ckpt /path/to/best_ckpt.pth \
  --num-sites 2 \
  --replay-npz /path/to/replay_data.npz \
  --save /tmp/compliance_dp_out.npz
```

Dummy mode:

```bash
python examples/run_policy.py --policy compliance_dp --robot toddlerbot -- \
  --ckpt /path/to/best_ckpt.pth \
  --num-sites 2
```

Replay file keys:

- required image key: `image` / `images` / `rgb` / `camera`
- required pose key: `x_obs` / `pose` / `ee_pose`
- optional wrench key: `x_wrench` / `wrench` / `wrenches`
- optional motor key: `motor_pos` / `obs_motor_pos` / `qpos`
