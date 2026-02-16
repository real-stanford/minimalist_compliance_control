"""Vision model helpers for standalone diffusion policy."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torchvision


def get_resnet(name: str, weights=None, input_channels: int = 3, **kwargs) -> nn.Module:
    """Build ResNet and replace final FC with Identity."""
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    if input_channels != 3:
        original_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )
        if weights is not None and input_channels == 1:
            with torch.no_grad():
                resnet.conv1.weight.data = original_conv1.weight.data.mean(
                    dim=1, keepdim=True
                )

    resnet.fc = nn.Identity()
    return resnet


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Replace all submodules matching predicate."""
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]

    for *parent, key in bn_list:
        parent_module = root_module
        if parent:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src = parent_module[int(key)]
            parent_module[int(key)] = func(src)
        else:
            src = getattr(parent_module, key)
            setattr(parent_module, key, func(src))

    remaining = [
        k for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)
    ]
    assert len(remaining) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int = 16,
) -> nn.Module:
    """Swap BatchNorm2d layers with GroupNorm layers."""
    return replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=max(1, x.num_features // features_per_group),
            num_channels=x.num_features,
        ),
    )
