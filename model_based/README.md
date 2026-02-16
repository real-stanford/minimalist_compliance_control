# Model-Based Example

## Purpose

This example runs a standalone model-based compliance policy with the phase flow:
`prep -> kneel -> approach -> model_based`.

## Usage

Run policy:

```bash
python examples/run_toddlerbot_model_based.py
```

On macOS MuJoCo viewer:

```bash
mjpython examples/run_toddlerbot_model_based.py
```

Keyboard control (run in another terminal):

```bash
python model_based/keyboard_control_ball.py --host 127.0.0.1 --port 5592
```

Keyboard commands:

- `c`: reverse direction
- `l`: left-hand mode
- `r`: right-hand mode
- `b`: both-hands mode

Optional extra dependencies:

```bash
pip install qpsolvers osqp sympy pyzmq
```
