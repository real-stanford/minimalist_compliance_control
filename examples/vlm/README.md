# Compliance VLM Example

## Purpose

This example runs a standalone VLM-guided compliance policy with mode switching
(`waiting`, `wiping`, `drawing`) and outputs package-compatible `command_matrix`.

## Usage

Install dependencies:

```bash
pip install numpy scipy opencv-python joblib pyzmq requests open3d pycocotools
```

Set API key for predictor backend:

```bash
export GOOGLE_API_KEY=...
# or export OPENAI_API_KEY=...
```

Place depth calibration files in `examples/vlm/depth/params/`:

- `calibration.pkl`
- `rectification.npz`

Replay mode:

```bash
python -m examples.vlm.run_compliance_vlm \
  --robot-name toddlerbot_2xm \
  --mode drawing \
  --object "star" \
  --replay-npz /path/to/replay_data.npz \
  --save /tmp/compliance_vlm_out.npz
```

Dummy mode:

```bash
python -m examples.vlm.run_compliance_vlm \
  --robot-name toddlerbot_2xm \
  --mode waiting
```

Replay file keys:

- required left image: `left_image` / `image` / `images` / `rgb` / `camera`
- required pose: `x_obs` / `pose` / `ee_pose`
- optional right image: `right_image` / `image_right` / `right`
- optional wrench: `x_wrench` / `wrench` / `wrenches`
- optional head position: `head_pos` / `head_position` / `head_position_world`
- optional head quaternion (wxyz): `head_quat` / `head_quaternion` / `head_quaternion_world_wxyz`
