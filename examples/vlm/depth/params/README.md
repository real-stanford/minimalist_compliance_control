# Depth Calibration Files

Place the stereo calibration files here for VLM affordance prediction:

- `calibration.pkl`
- `rectification.npz`

`examples/vlm/affordance/affordance_predictor.py` loads these by default.
You can also pass custom paths via `AffordancePredictor(depth_config=...)`.
