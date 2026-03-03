"""Standalone diffusion policy inference model.

This is adapted from toddlerbot_internal's DPModel with local-only imports.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np

from .models.diffusion_model import ConditionalUnet1D
from .utils.dataset_utils import normalize_data, unnormalize_data

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    models = None
    DDIMScheduler = None
    DDPMScheduler = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from .utils.model_utils import get_resnet, replace_bn_with_gn


class DPModel:
    """Diffusion Policy model for manipulation inference."""

    def __init__(
        self,
        ckpt_path: str,
        stats: Optional[dict] = None,
        diffuse_steps: int = 3,
        action_horizon: Optional[int] = None,
        use_ddpm: bool = True,
    ) -> None:
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError(
                "DPModel requires torch/torchvision/diffusers. "
                "Install them to run diffusion policy inference."
            ) from _IMPORT_ERROR

        assert torch is not None
        assert models is not None
        assert DDPMScheduler is not None
        assert DDIMScheduler is not None

        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

        self.device = torch.device(device_str)
        self.diffuse_steps = int(diffuse_steps)
        if self.diffuse_steps < 1:
            raise ValueError("diffuse_steps must be >= 1")
        self.use_ddpm = bool(use_ddpm)

        ckpt = self.load_checkpoint(ckpt_path)
        params = ckpt["params"]
        self.params = params

        self.weights = (
            None
            if len(params.get("weights", "")) == 0
            else models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.vision_feature_dim = params["vision_feature_dim"]
        self.lowdim_obs_dim = params["lowdim_obs_dim"]
        self.action_dim = params["action_dim"]
        self.input_channels = params.get("input_channels")
        self.pred_horizon = params["pred_horizon"]
        self.obs_horizon = params["obs_horizon"]
        self.image_horizon = params.get("image_horizon", self.obs_horizon)
        if action_horizon is None:
            self.action_horizon = self.pred_horizon - self.obs_horizon + 1
        else:
            self.action_horizon = int(action_horizon)
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")

        self.obs_dim = self.vision_feature_dim + self.lowdim_obs_dim
        self.cond_dim = (
            self.vision_feature_dim * self.image_horizon
            + self.lowdim_obs_dim * self.obs_horizon
        )

        self.num_diffusion_iters = 100
        self.noise_scheduler_ddpm = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.noise_scheduler_ddim = DDIMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
            timestep_spacing="linspace",
        )

        self.down_dims = [128, 256, 384]
        self.load_model(ckpt_path, stats=stats, ckpt=ckpt)

    def load_checkpoint(self, ckpt_path: str) -> dict:
        assert torch is not None
        try:
            return torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location=self.device)

    def load_model(
        self, ckpt_path: str, stats: Optional[dict] = None, ckpt: Optional[dict] = None
    ) -> None:
        assert torch is not None
        assert nn is not None

        if ckpt is None:
            ckpt = self.load_checkpoint(ckpt_path)
        state_dict = ckpt["state_dict"]

        if self.input_channels is None:
            conv_weight = state_dict.get("vision_encoder.conv1.weight")
            if conv_weight is not None:
                self.input_channels = int(conv_weight.shape[1])
            else:
                self.input_channels = 1

        weights = self.weights if self.input_channels in (1, 3) else None
        vision_encoder = get_resnet(
            "resnet18", weights=weights, input_channels=self.input_channels
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.cond_dim,
            down_dims=self.down_dims,
        )

        self.ema_nets = nn.ModuleDict(
            {
                "vision_encoder": vision_encoder,
                "noise_pred_net": noise_pred_net,
            }
        ).to(self.device)

        if stats is None:
            self.ema_nets.load_state_dict(state_dict)
            self.stats = ckpt["stats"]
        else:
            self.stats = stats
            self.ema_nets.load_state_dict(state_dict)

        self.ema_nets.eval()

    def prepare_inputs(
        self,
        obs_deque: deque,
        image_deque: Optional[deque] = None,
    ) -> tuple[Any, Any]:
        assert torch is not None

        if image_deque is None:
            images = np.stack([x["image"] for x in obs_deque])
        else:
            images = np.stack(list(image_deque))

        if images.shape[0] > self.image_horizon:
            images = images[-self.image_horizon :]
        if images.shape[0] < self.image_horizon:
            pad = np.repeat(images[-1:], self.image_horizon - images.shape[0], axis=0)
            images = np.concatenate([images, pad], axis=0)

        if image_deque is None:
            obs_values = np.stack([x["obs"] for x in obs_deque])
        else:
            obs_values = np.stack(list(obs_deque))

        if obs_values.shape[0] > self.obs_horizon:
            obs_values = obs_values[-self.obs_horizon :]
        elif obs_values.shape[0] < self.obs_horizon:
            pad = np.repeat(
                obs_values[-1:], self.obs_horizon - obs_values.shape[0], axis=0
            )
            obs_values = np.concatenate([obs_values, pad], axis=0)

        nobs = normalize_data(obs_values, stats=self.stats["obs"])
        nimages = images

        nimages = torch.from_numpy(nimages).to(self.device, dtype=torch.float32)
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)
        return nimages, nobs

    def prediction_to_action(self, naction: Any) -> np.ndarray:
        naction = naction.detach().to("cpu").numpy()[0]
        action_pred = unnormalize_data(naction, stats=self.stats["action"])
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        return action_pred[start:end, :]

    def inference_ddim(
        self, obs_cond: Any, nsteps: int = 10, naction: Optional[Any] = None
    ) -> Any:
        assert torch is not None
        if naction is None:
            naction = torch.randn(
                (1, self.pred_horizon, self.action_dim), device=self.device
            )

        self.noise_scheduler_ddim.set_timesteps(nsteps)
        for k in self.noise_scheduler_ddim.timesteps:
            noise_pred = self.ema_nets["noise_pred_net"](
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )
            naction = self.noise_scheduler_ddim.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

        return naction

    def inference_ddpm(
        self, obs_cond: Any, nsteps: int, naction: Optional[Any] = None
    ) -> Any:
        assert torch is not None
        if naction is None:
            naction = torch.randn(
                (1, self.pred_horizon, self.action_dim), device=self.device
            )

        self.noise_scheduler_ddpm.set_timesteps(nsteps)
        for k in self.noise_scheduler_ddpm.timesteps:
            noise_pred = self.ema_nets["noise_pred_net"](
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )
            naction = self.noise_scheduler_ddpm.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample

        return naction

    def get_action_from_obs(
        self,
        obs_deque: deque,
        image_deque: Optional[deque] = None,
    ) -> np.ndarray:
        assert torch is not None
        nimages, nobs = self.prepare_inputs(obs_deque, image_deque=image_deque)

        with torch.no_grad():
            image_features = self.ema_nets["vision_encoder"](nimages)
            image_flat = image_features.reshape(1, -1)
            obs_flat = nobs.reshape(1, -1)
            obs_cond = torch.cat([image_flat, obs_flat], dim=-1)

            if self.use_ddpm:
                naction = self.inference_ddpm(obs_cond, nsteps=self.diffuse_steps)
            else:
                naction = self.inference_ddim(obs_cond, nsteps=self.diffuse_steps)

        return self.prediction_to_action(naction)
