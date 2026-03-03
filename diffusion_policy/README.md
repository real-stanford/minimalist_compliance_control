# Compliance DP Components

## Purpose

This folder contains diffusion-policy model components used by
`policy/compliance_dp.py`. The policy outputs `pose_command` and
`command_matrix` compatible with the compliance controller layout.

## Required Dependencies

```bash
pip install torch torchvision diffusers opencv-python joblib
```

## Replay Input Conventions

Accepted replay keys:

- required image key: `image` / `images` / `rgb` / `camera`
- required pose key: `x_obs` / `pose` / `ee_pose`
- optional wrench key: `x_wrench` / `wrench` / `wrenches`
- optional motor key: `motor_pos` / `obs_motor_pos` / `qpos`
