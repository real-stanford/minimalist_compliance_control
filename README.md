# minimalist_compliance_control

A lightweight package for MuJoCo-based compliance control and wrench estimation.

## Overview

`minimalist_compliance_control` provides a minimal, reusable control stack with:

- wrench simulation and Jacobian utilities,
- online wrench estimation,
- compliance reference integration,
- a unified controller interface.

## Installation

```bash
conda create -n mcc python=3.10
conda activate mcc
pip install -e .
```

For example stacks (model-based / diffusion-policy / VLM):

```bash
pip install -e ".[examples]"
```

## Core Modules

- `minimalist_compliance_control/wrench_sim.py`
- `minimalist_compliance_control/wrench_estimation.py`
- `minimalist_compliance_control/controller.py`
- `minimalist_compliance_control/reference/`

## Examples

Example usage is documented under:

- `model_based/`
- `diffusion_policy/`
- `vlm/`
