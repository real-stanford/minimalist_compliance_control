# LEAP-Hand Minimal Policy

This folder contains a standalone copy of toddlerbot's `leap_rotate_compliance` logic,
refactored to avoid `toddlerbot` runtime dependencies.

## Included

- `leap_rotate_compliance.py`
  - standalone `LeapRotateCompliancePolicy`
  - same close/rotate state machine and OCHS-based action distribution logic
  - keyboard commands: `c` (reverse), `r` (switch mode)
- `hybrid_servo/demo/multi_finger_rotate_anything/ochs_helpers.py`
  - copied helper functions used by LEAP policy

## External dependencies

This module depends only on packages already used by `examples/model_based_minimal`:

- `numpy`
- `scipy`
- `mujoco`
- `sympy`
- `qpsolvers` + `osqp` (for OCHS solver)
- `pyzmq` (optional, for keyboard control)

## Run (Self-Contained)

```bash
python examples/model_based_minimal/leaphand/leap_rotate_compliance.py
```

On macOS MuJoCo viewer:

```bash
mjpython examples/model_based_minimal/leaphand/leap_rotate_compliance.py
```

Useful flags:

- `--headless`
- `--duration 60`
- `--scene-xml examples/descriptions/leap_hand_rotation/scene_fixed.xml`
- `--keyboard-port 5592`

## Usage sketch

```python
from leaphand.leap_rotate_compliance import LeapRotateCompliancePolicy

policy = LeapRotateCompliancePolicy(
    wrench_sim=wrench_sim,
    wrench_site_names=("if_tip", "mf_tip", "th_tip"),
    control_dt=0.02,
    prep_duration=0.0,
)

outputs = policy.step(
    time_curr=sim_time,
    wrenches_by_site=wrench_dict,
)

pose_command = outputs["pose_command"]
wrench_command = outputs["wrench_command"]
```

`policy.step(...)` returns a dict with:
- `phase`
- `control_mode`
- `pose_command`
- `wrench_command`
- `pos_stiffness`, `rot_stiffness`
- `pos_damping`, `rot_damping`
