# LEAP-Hand Example

## Purpose

This example runs a standalone LEAP-hand compliance policy for approach, contact,
and OCHS-based rotation/translation control.

## Usage

Run:

```bash
python examples/model_based/leaphand/leap_rotate_compliance.py
```

On macOS MuJoCo viewer:

```bash
mjpython examples/model_based/leaphand/leap_rotate_compliance.py
```

Common options:

- `--headless`
- `--duration 60`
- `--scene-xml examples/descriptions/leap_hand_rotation/scene_fixed.xml`
- `--keyboard-port 5592`

Optional extra dependencies:

```bash
pip install qpsolvers osqp sympy pyzmq
```
