# Model-Based Example

## Purpose

This example runs a standalone model-based compliance policy with the phase flow:
`prep -> kneel -> approach -> model_based`.

## Usage

Run policy:

```bash
python examples/model_based/toddlerbot/run_model_based_minimal.py
```

On macOS MuJoCo viewer:

```bash
mjpython examples/model_based/toddlerbot/run_model_based_minimal.py
```

Keyboard control (run in another terminal):

```bash
python examples/model_based/utils/keyboard_control_ball.py --host 127.0.0.1 --port 5592
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
