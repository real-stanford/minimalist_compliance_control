# Compliance DP (Self-Contained)

This folder ports toddlerbot_internal `compliance_dp` inference core into a standalone example.

## Included

- `compliance_dp.py`
  - `StandaloneComplianceDP`: async diffusion inference + action timing + pose-command conversion
  - outputs `pose_command` and `command_matrix` compatible with
    `minimalist_compliance_control.reference.compliance_ref.COMMAND_LAYOUT`
- `dp_model.py`
  - local `DPModel` adapted from `toddlerbot/manipulation/inference_class.py`
- `models/diffusion_model.py`
  - local `ConditionalUnet1D`
- `utils/`
  - normalization, model helpers, trajectory interpolation
- `run_compliance_dp.py`
  - replay runner for offline data (`.npz`) or dummy input

## Dependencies

In addition to repo defaults, compliance DP inference requires:

```bash
pip install torch torchvision diffusers opencv-python joblib
```

## Run

Replay mode (recommended):

```bash
python -m examples.diffusion_policy.run_compliance_dp \
  --ckpt /path/to/best_ckpt.pth \
  --num-sites 2 \
  --replay-npz /path/to/replay_data.npz \
  --save /tmp/compliance_dp_out.npz
```

Dummy mode:

```bash
python -m examples.diffusion_policy.run_compliance_dp \
  --ckpt /path/to/best_ckpt.pth \
  --num-sites 2
```

## Replay file keys

Required keys:
- image (`image`/`images`/`rgb`/`camera`)
- x_obs (`x_obs`/`pose`/`ee_pose`)

Optional keys:
- x_wrench (`x_wrench`/`wrench`/`wrenches`)
- motor_pos (`motor_pos`/`obs_motor_pos`/`qpos`)

## Notes

- This migration targets `compliance_dp` behavior (not the old `dp_policy` motor-only path).
- The module is self-contained and does not import `toddlerbot.*` runtime classes.
