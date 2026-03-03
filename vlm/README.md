# Compliance VLM Components

## Purpose

This folder contains VLM affordance/depth/server components used by
`policy/compliance_vlm.py` for mode-driven compliance (`waiting`, `wiping`,
`drawing`).

## Dependencies

```bash
pip install numpy scipy opencv-python joblib pyzmq requests open3d pycocotools
```

## Provider Keys

Set one of:

- `GOOGLE_API_KEY`
- `OPENAI_API_KEY`

## Camera Config Assets

Place camera YAML files in `assets/`:

- `toddlerbot_camera.yml`
- `leap_camera.yml`

## Replay Input Conventions

Accepted replay keys:

- required left image: `left_image` / `image` / `images` / `rgb` / `camera`
- required pose: `x_obs` / `pose` / `ee_pose`
- optional right image: `right_image` / `image_right` / `right`
- optional wrench: `x_wrench` / `wrench` / `wrenches`
- optional head position: `head_pos` / `head_position` / `head_position_world`
- optional head quaternion (wxyz): `head_quat` / `head_quaternion` / `head_quaternion_world_wxyz`
