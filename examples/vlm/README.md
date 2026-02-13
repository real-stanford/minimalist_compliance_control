# Compliance VLM (Self-Contained)

This folder ports `toddlerbot_internal` `compliance_vlm` policy into a standalone module.

## Included

- `compliance_vlm.py`
  - `StandaloneComplianceVLM`: mode switching (`waiting/wiping/drawing`), async affordance prediction, trajectory execution, fixed-trajectory replay, command matrix output.
- `run_compliance_vlm.py`
  - replay/dummy runner for offline testing.
- `affordance/`
  - local copies of affordance predictor + trajectory planner modules.
- `depth/`
  - local depth utility + rectifier modules.
- `utils/`
  - local camera/zmq/math helpers.

## Dependencies

```bash
pip install numpy scipy opencv-python joblib pyzmq requests open3d pycocotools
```

For real affordance prediction, also set API key and ensure foundation/SAM services are reachable:

```bash
export GOOGLE_API_KEY=...   # for gemini provider
# or export OPENAI_API_KEY=... if using openai provider
```

## Depth Calibration Files

Put these files under `examples/vlm/depth/params/`:

- `calibration.pkl`
- `rectification.npz`

## Run

Replay mode (recommended):

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

## Replay Keys

Required keys:

- left image: `left_image` / `image` / `images` / `rgb` / `camera`
- `x_obs`: `x_obs` / `pose` / `ee_pose`

Optional keys:

- right image: `right_image` / `image_right` / `right`
- `x_wrench`: `x_wrench` / `wrench` / `wrenches`
- head position: `head_pos` / `head_position` / `head_position_world`
- head quaternion (wxyz): `head_quat` / `head_quaternion` / `head_quaternion_world_wxyz`

## Notes

- Output command matrix is aligned with `minimalist_compliance_control.reference.compliance_ref.COMMAND_LAYOUT`.
- This migration is self-contained and does not import `toddlerbot.*` runtime modules.
