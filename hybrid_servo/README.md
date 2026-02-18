# Model-Based Example

## Purpose

This example runs a standalone model-based compliance policy with the phase flow:
`prep -> kneel -> approach -> model_based`.

## Usage

Run policy:

```bash
python examples/run_policy.py --policy compliance_model_based --robot toddlerbot
```

On macOS MuJoCo viewer:

```bash
mjpython examples/run_policy.py --policy compliance_model_based --robot toddlerbot
```

Keyboard control (focus the same terminal running the policy):

Keyboard commands:

- `c`: reverse direction
- `l`: left-hand mode
- `r`: right-hand mode
- `b`: both-hands mode

Optional extra dependencies:

```bash
pip install qpsolvers osqp sympy
```
