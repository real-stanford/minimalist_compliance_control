# Codex Workspace Notes

## Scope Covered
- Read all repository text/source/config files under `minimalist_compliance_control/`, `examples/`, and root metadata files.
- Excluded binary artifacts from deep inspection (`.stl`, `.obj`, `.png`, `.pack`, `.idx`, `.lz4`, `.glb`).

## Project Goal
A lightweight MuJoCo compliance-control package with:
- wrench simulation (`WrenchSim`),
- wrench estimation (`estimate_wrench`),
- compliance reference integration (`ComplianceReference` + Mink IK),
- end-to-end controller orchestration (`ComplianceController`).

## Core Package Map
- `minimalist_compliance_control/wrench_sim.py`
  - MuJoCo wrapper for loading XML, setting state, Jacobian extraction, bias torque, rendering/viewer, recording.
  - Important: `site_jacobian`, `bias_torque`, `set_qpos`, `set_joint_positions`, `set_dof_positions`.
- `minimalist_compliance_control/wrench_estimation.py`
  - Dense/axis-projected wrench recovery from Jacobians and external torque estimate.
  - `WrenchEstimateConfig` controls force/torque regularization and axis behavior.
- `minimalist_compliance_control/compliance_ref.py`
  - Defines `COMMAND_LAYOUT` (54-wide command matrix slices).
  - Integrates pose/velocity references and calls IK (`MinkIK`) to produce actuator refs.
- `minimalist_compliance_control/ik_solvers.py`
  - `MinkIK` builds Mink posture + site tasks and solves iterative IK.
- `minimalist_compliance_control/controller.py`
  - Pipeline: state sync -> per-site wrench estimation -> optional measured wrench injection into command matrix -> optional compliance state update.

## Data Contract
`COMMAND_LAYOUT` slices (54 values per site):
- desired position/orientation,
- measured force/torque,
- stiffness/damping matrices (`kp/kd` for pos/rot, flattened 3x3),
- commanded force/torque.

This layout is the shared interface across core, model-based, diffusion-policy, and VLM examples.

## Example Families
- `examples/`
  - Runner scripts for toddlerbot/leaphand compliance and top-level launchers.
- `model_based/`
  - OCHS/HFVC-based policies.
  - Major scripts are large monoliths:
    - `examples/run_toddlerbot_model_based.py` (~1873 lines)
    - `examples/run_leap_model_based.py` (~1878 lines)
  - Includes local copy of hybrid-servo algorithms under `model_based/hybrid_servo/`.
- `diffusion_policy/`
  - Diffusion inference wrapper (`StandaloneComplianceDP`) and model blocks.
- `vlm/`
  - VLM-guided compliance policy (`StandaloneComplianceVLM`) with affordance + depth + communication utilities.

## Config and Robot Assets
- Gin configs:
  - `config/toddlerbot.gin`
  - `config/leap.gin`
  - `config/toddlerbot_model_based.gin`
- Robot/environment descriptions under `descriptions/`:
  - toddlerbot XML/URDF/YAML variants,
  - leap-hand XML + object XMLs,
  - default actuator parameter YAML.

## Dependencies
From `pyproject.toml` core deps:
- `numpy`, `scipy`, `mujoco`, `mink`, `gin-config`.

Optional example deps:
- model-based: `qpsolvers`, `osqp`, `sympy`, `pyzmq` (and sometimes `cvxopt` path in `solvehfvc.py`),
- diffusion policy: `torch`, `torchvision`, `diffusers`, `opencv-python`, `joblib`,
- VLM: `requests`, `open3d`, `pycocotools`, `pyzmq`, plus provider API key env vars.

## Notable Risks / Gotchas
- `ComplianceController.step()` calls `self.wrench_sim.set_motor_angles(motor_pos)` when no `qpos/joint_pos` is provided, but `WrenchSim` currently does not define `set_motor_angles`.
  - This path may fail at runtime unless callers always pass `qpos` or `joint_pos`.
- Several example scripts are very large and multi-responsibility; future edits should isolate helper logic first before behavior changes.
- Many MuJoCo XML variants are near-duplicates (fixed, mjx, pos, model-based), so sync drift risk is high.

## Practical Entry Points
- Core package install: `pip install -e .`
- Basic examples:
  - `python examples/run_compliance.py --robot toddlerbot`
  - `python examples/run_compliance.py --robot leap`
- Model-based:
  - `python examples/run_toddlerbot_model_based.py`
  - `python examples/run_leap_model_based.py`
- Diffusion policy:
  - `python examples/run_compliance_dp.py ...`
- VLM:
  - `python examples/run_compliance_vlm.py ...`

## Editing Guidance For Future Codex Turns
- Preserve `COMMAND_LAYOUT` compatibility unless all consumers are updated.
- When touching controller estimation, verify site-to-joint and site-to-motor index mappings in Gin.
- For physics behavior changes, validate both fixed-base and floating-base configurations.
- Treat `descriptions/*` as source-of-truth robot assets; avoid ad-hoc one-off XML divergence.
