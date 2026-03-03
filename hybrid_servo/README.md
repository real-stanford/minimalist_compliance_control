# Model-Based Components

## Purpose

This folder contains OCHS/HFVC model-based components used by the model-based
compliance policy (`prep -> kneel -> approach -> model_based -> mode switching`).

## Integration Notes

- Primary integration target:
  - `policy/compliance_model_based_toddlerbot.py`
  - `policy/compliance_model_based_leap.py`
- Keyboard commands used by the model-based policies:
  - `c`: reverse direction
  - `l`: left-hand mode
  - `r`: right-hand mode
  - `b`: both-hands mode

## Optional Dependencies

```bash
pip install qpsolvers osqp sympy
```
