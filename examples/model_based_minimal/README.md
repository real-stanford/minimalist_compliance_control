# Minimal Model-Based Example

This folder contains a standalone, minimal model-based policy demo.

## What is included

- `config.gin`: isolated config for this example only (aligned with toddlerbot_internal values for `dt`, IK/Mink settings, actuator/joint mapping).
- `run_model_based_minimal.py`: policy runtime loop with three phases:
  - `kneel` trajectory replay
  - `approach` to ball contact
  - `model_based` OCHS + compliance control
- `utils/kneel_2xm.lz4`: kneel motion used by the first phase.
- `utils/keyboard_control_ball.py`: keyboard command sender (`c/l/r/b`).
- `utils/zmq_control.py`: minimal ZMQ keyboard sender/receiver utilities.
- `hybrid_servo/algorithm/ochs.py`: copied OCHS solver.
- `hybrid_servo/demo/two_hand_rotate_ball/ochs_helpers.py`: copied OCHS input/jacobian helper functions.
- `hybrid_servo/demo/multi_finger_rotate_anything/ochs_helpers.py`: copied OCHS helpers for LEAP rotate compliance.
- `leaphand/`: standalone LEAP rotate compliance policy + OCHS helpers copied from toddlerbot.

## Run

```bash
python examples/model_based_minimal/toddlerbot/run_model_based_minimal.py
```

On macOS MuJoCo viewer, use:

```bash
mjpython examples/model_based_minimal/toddlerbot/run_model_based_minimal.py
```

In another terminal, send keyboard commands:

```bash
python examples/model_based_minimal/utils/keyboard_control_ball.py --host 127.0.0.1 --port 5592
```

Commands:

- `c`: reverse rotation direction
- `l`: switch to left-hand mode
- `r`: switch to right-hand mode
- `b`: switch to both-hands mode

Mode goals follow toddlerbot_internal:

- `left`: axis = `[1, 0, 0]`
- `right`: axis = `[-1, 0, 0]`
- `both`: axis = `[0, 0, 1]`

If needed, install extra dependencies for this example:

```bash
pip install qpsolvers osqp sympy pyzmq
```

## Notes

- This is intentionally minimal and independent from `toddlerbot_internal` runtime classes.
- OCHS and helper logic are directly copied into this folder for isolation.
- It uses `examples/descriptions/toddlerbot_2xm/scene_ball.xml` as the simulation scene.
- It does not include DP/VLM policy logic or hand-switching state machine.
