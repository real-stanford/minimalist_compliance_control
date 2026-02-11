# minimum_compliance

Minimal compliance control + wrench estimation core with a clean interface.

## Package layout
- `wrench_sim.py`: standalone MuJoCo backend for Jacobians/bias torques
- `wrench_estimation.py`: Jacobian-based wrench estimation utilities
- `controller.py`: unified pipeline controller
- `reference/`: compliance reference utilities (Mink/MuJoCo/MJX/JAX)

## Install (editable)

```bash
pip install -e minimum_compliance
```

## Usage

Import the controller, sim, and reference modules as needed. This package does not
ship example scripts after cleanup.
