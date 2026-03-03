"""Train diffusion policy models for robot manipulation tasks."""

import argparse
import json
import os
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from diffusion_policy.models.diffusion_model import ConditionalUnet1D
from diffusion_policy.teleop_dataset import TeleopImageDataset
from diffusion_policy.utils.model_utils import get_resnet, replace_bn_with_gn


class Tee:
    """Mirror stdout/stderr to a log file."""

    def __init__(self, log_path: str) -> None:
        self.terminal = sys.__stdout__
        self.log = open(log_path, "w", buffering=1, encoding="utf-8")
        self.closed = False

    def write(self, message: str) -> None:
        msg = str(message)
        try:
            self.terminal.write(msg)
            self.terminal.flush()
            if not self.closed:
                self.log.write(msg)
                self.log.flush()
        except (ValueError, OSError):
            self.closed = True

    def flush(self) -> None:
        try:
            self.terminal.flush()
            if not self.closed:
                self.log.flush()
        except (ValueError, OSError):
            self.closed = True

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()

    def close(self) -> None:
        if not self.closed:
            try:
                self.log.close()
            except Exception:
                pass
            self.closed = True


def train(
    dataset_path_list: List[str],
    exp_folder_path: str,
    weights: str,
    pred_horizon: int,
    obs_horizon: int,
    obs_source: List[str] | None = None,
    image_views: str = "left",
    image_horizon: int = 1,
    action_source: List[str] | None = None,
    proprio_noise_std: float = 0.0,
    action_dim: int | None = None,
    vision_feature_dim: int = 512,
    num_diffusion_iters: int = 100,
    batch_size: int = 256,
    restore: str | None = None,
    num_epochs: int = 1000,
    early_stopping_patience: int = 50,  # Stop if no improvement for X epochs
    train_split_ratio: float = 0.9,
):
    """Trains a neural network model using a dataset of teleoperation images and actions.

    Args:
        dataset_path_list (List[str]): List of paths to the datasets.
        exp_folder_path (str): Path to the folder where experiment outputs will be saved.
        weights (str): Pre-trained weights for the vision encoder.
        pred_horizon (int): Prediction horizon for the model.
        obs_horizon (int): Observation horizon for the model.
        obs_source (List[str], optional): Observation keys to use. Defaults to None.
        image_views (str, optional): Image views to load: left, right, or both. Defaults to "left".
        image_horizon (int, optional): Number of image frames to keep. Defaults to 1.
        action_source (List[str], optional): Action key to use. Defaults to None.
        proprio_noise_std (float, optional): Stddev of Gaussian noise for proprio. Defaults to 0.0.
        action_dim (int, optional): Dimensionality of the action space. Defaults to None.
        vision_feature_dim (int, optional): Dimensionality of the vision feature space. Defaults to 512.
        num_diffusion_iters (int, optional): Number of diffusion iterations. Defaults to 100.
        batch_size (int, optional): Batch size for training/validation. Defaults to 256.
        restore (str, optional): Path to a checkpoint to restore weights from. Defaults to None.
        num_epochs (int, optional): Number of training epochs. Defaults to 1000.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement before stopping early. Defaults to 100.
        train_split_ratio (float, optional): Ratio of the dataset to use for training. Defaults to 0.8.
    """
    plt.switch_backend("Agg")

    # ### **Network Demo**
    if image_horizon < 1:
        raise ValueError("image_horizon must be >= 1.")
    dataset = TeleopImageDataset(
        dataset_path_list,
        exp_folder_path,
        pred_horizon,
        obs_horizon,
        obs_source=obs_source,
        image_views=image_views,
        image_horizon=image_horizon,
        action_source=action_source,
    )

    # Detect number of channels from the dataset
    sample_batch = dataset[0]
    input_channels = sample_batch["image"].shape[1]  # C dimension in (C, H, W)
    print(f"Detected {input_channels} input channels")
    dataset_action_dim = sample_batch["action"].shape[-1]
    dataset_obs_dim = sample_batch["obs"].shape[-1]
    if action_dim is None:
        action_dim = dataset_action_dim
    elif action_dim != dataset_action_dim:
        print(
            f"Warning: action_dim={action_dim} does not match dataset "
            f"({dataset_action_dim}). Using dataset value."
        )
        action_dim = dataset_action_dim

    # Split into train/val sets
    train_size = int(len(dataset) * train_split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    sample_batch = next(iter(train_dataloader))
    print(
        "Sample batch shapes:",
        {key: value.shape for key, value in sample_batch.items()},
    )

    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet(
        "resnet18",
        weights=None if len(weights) == 0 else models.ResNet18_Weights.IMAGENET1K_V1,
        input_channels=input_channels,
    )
    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    lowdim_obs_dim = dataset_obs_dim
    cond_dim = vision_feature_dim * image_horizon + lowdim_obs_dim * obs_horizon
    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=cond_dim,
        down_dims=[128, 256, 384],
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict(
        {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
    )

    if restore:
        ckpt = torch.load(restore, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = nets.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Warning: restore missing keys: {missing}")
            print(f"Warning: restore unexpected keys: {unexpected}")

    print(
        "ve # weights: ",
        np.sum([param.nelement() for param in vision_encoder.parameters()]),
    )
    print(
        "unet # weights: ",
        np.sum([param.nelement() for param in noise_pred_net.parameters()]),
    )

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule="squaredcos_cap_v2",
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type="epsilon",
    )

    # device transfer
    device = torch.device("cuda")
    _ = nets.to(device)

    # ### **Training**
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    params = {
        "weights": weights,
        "pred_horizon": pred_horizon,
        "obs_horizon": obs_horizon,
        "image_horizon": image_horizon,
        "vision_feature_dim": vision_feature_dim,
        "lowdim_obs_dim": lowdim_obs_dim,
        "action_dim": action_dim,
        "input_channels": input_channels,
        "batch_size": batch_size,
        "obs_source": obs_source,
        "image_views": image_views,
        "action_source": action_source,
        "proprio_noise_std": proprio_noise_std,
        "restore": restore,
    }
    print(params)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses_per_epoch = []
    val_losses_per_epoch = []

    tglobal = tqdm(range(num_epochs), desc="Epoch")
    try:
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(train_dataloader, desc="Batch", leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nimage = nbatch["image"][:, :image_horizon].to(
                        device, dtype=torch.float32
                    )
                    nobs = nbatch["obs"][:, :obs_horizon].to(
                        device, dtype=torch.float32
                    )
                    if proprio_noise_std > 0.0:
                        nobs = nobs + torch.randn_like(nobs) * proprio_noise_std
                    naction = nbatch["action"].to(device, dtype=torch.float32)
                    B = nobs.shape[0]

                    # encoder vision features
                    image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage.shape[:2], -1)
                    # (B,image_horizon,D)
                    image_features = image_features.flatten(start_dim=1)

                    # concatenate vision feature and low-dim obs
                    obs_features = nobs.flatten(start_dim=1)
                    obs_cond = torch.cat([image_features, obs_features], dim=-1)
                    # (B, image_horizon * vision_feature_dim + obs_horizon * lowdim_obs_dim)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (B,),
                        device=device,
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond
                    )

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            mean_train_loss = np.mean(epoch_loss)
            tglobal.set_postfix(train_loss=mean_train_loss)
            train_losses_per_epoch.append(mean_train_loss)

            # Validation loop
            nets.eval()
            val_losses = []
            with torch.no_grad():
                for nbatch in val_dataloader:
                    nimage = nbatch["image"][:, :image_horizon].to(
                        device, dtype=torch.float32
                    )
                    nobs = nbatch["obs"][:, :obs_horizon].to(
                        device, dtype=torch.float32
                    )
                    naction = nbatch["action"].to(device, dtype=torch.float32)
                    B = nobs.shape[0]

                    image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage.shape[:2], -1)
                    image_features = image_features.flatten(start_dim=1)

                    obs_features = nobs.flatten(start_dim=1)
                    obs_cond = torch.cat([image_features, obs_features], dim=-1)

                    noise = torch.randn(naction.shape, device=device)

                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (B,),
                        device=device,
                    ).long()

                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    noise_pred = nets["noise_pred_net"](
                        noisy_actions, timesteps, global_cond=obs_cond
                    )

                    val_loss = nn.functional.mse_loss(noise_pred, noise)
                    val_losses.append(val_loss.item())

            mean_val_loss = np.mean(val_losses)
            tglobal.set_postfix(train_loss=mean_train_loss, val_loss=mean_val_loss)
            val_losses_per_epoch.append(mean_val_loss)

            # Early stopping check
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss.item()
                patience_counter = 0
                # Update EMA nets if this is the best so far
                best_ema_nets = nn.ModuleDict(
                    {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
                )
                ema.copy_to(best_ema_nets.parameters())
                # Save best model checkpoint immediately
                ckpt_path = os.path.join(exp_folder_path, "best_ckpt.pth")
                torch.save(
                    {
                        "state_dict": best_ema_nets.state_dict(),
                        "stats": dataset.stats,
                        "params": params,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

    except KeyboardInterrupt:
        pass

    # If we stopped early, best checkpoint is already saved.
    # If we completed all epochs without triggering early stopping, save final EMA weights.
    if patience_counter < early_stopping_patience:
        ema_nets = nets
        ema.copy_to(ema_nets.parameters())

        ckpt_path = os.path.join(exp_folder_path, "last_ckpt.pth")
        # Save final model checkpoint
        torch.save(
            {
                "state_dict": ema_nets.state_dict(),
                "stats": dataset.stats,
                "params": params,
            },
            ckpt_path,
        )

    print(f"Best validation loss: {best_val_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw data to create dataset.")
    parser.add_argument(
        "--robot",
        type=str,
        default="toddlerbot",
        help="The name of the robot. Need to match the name in descriptions.",
    )
    parser.add_argument(
        "--dataset-paths",
        type=str,
        default="",
        help="Space-separated dataset directories (dataset.lz4 is appended).",
    )
    parser.add_argument(
        "--pred-horizon",
        type=int,
        default=30,
        help="The horizon of the prediction.",
    )
    parser.add_argument(
        "--obs-horizon",
        type=int,
        default=1,
        help="The horizon of the low-dim observation.",
    )
    parser.add_argument(
        "--image-horizon",
        type=int,
        default=1,
        help="The horizon of image observations.",
    )
    parser.add_argument(
        "--obs-source",
        type=str,
        nargs="*",
        default=["x_obs"],
        help="Observation keys to use (space- or comma-separated).",
    )
    parser.add_argument(
        "--action-source",
        type=str,
        nargs="*",
        default=["x_cmd"],
        help="Action key to use (space- or comma-separated).",
    )
    parser.add_argument(
        "--proprio-noise-std",
        type=float,
        default=0.0,
        help="Stddev of Gaussian noise added to proprio during training.",
    )
    parser.add_argument(
        "--image-views",
        type=str.lower,
        choices=["left", "right", "both"],
        default="left",
        help=(
            "Image views to load: left, right, or both. "
            "Datasets must include left_image/right_image keys."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="The pretrained weights.",
    )
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="Path to a checkpoint to restore weights from.",
    )

    args = parser.parse_args()

    obs_source = None
    if args.obs_source:
        obs_source = [
            entry.strip()
            for token in args.obs_source
            for entry in token.split(",")
            if entry.strip()
        ] or None

    action_source = None
    if args.action_source:
        action_source = [
            entry.strip()
            for token in args.action_source
            for entry in token.split(",")
            if entry.strip()
        ] or None

    dataset_path_list = []
    dataset_tokens = [token for token in args.dataset_paths.split(" ") if token]
    for token in dataset_tokens:
        token_path = token.strip()
        if not token_path:
            continue
        if token_path.endswith(".lz4"):
            token_path = os.path.dirname(token_path)
        dataset_path_list.append(os.path.join(token_path, "dataset.lz4"))

    if not dataset_path_list:
        raise ValueError("No dataset paths resolved. Use --dataset-paths.")

    exp_name = f"{args.robot}_dp"
    time_str = time.strftime("%Y%m%d_%H%M%S")
    exp_folder_path = f"results/{exp_name}_{time_str}"
    os.makedirs(exp_folder_path, exist_ok=True)
    log_path = os.path.join(exp_folder_path, "train.log")
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee
    try:
        print(f"Experiment folder: {exp_folder_path}")
        print(f"Logging to {log_path}")
        args_path = os.path.join(exp_folder_path, "args.json")
        with open(args_path, "w", encoding="utf-8") as args_file:
            json.dump(
                {
                    **vars(args),
                    "dataset_path_list": dataset_path_list,
                    "obs_source": obs_source,
                    "action_source": action_source,
                },
                args_file,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
        print(f"Wrote args to {args_path}")
        train(
            dataset_path_list,
            exp_folder_path,
            args.weights,
            args.pred_horizon,
            args.obs_horizon,
            obs_source=obs_source,
            image_views=args.image_views,
            image_horizon=args.image_horizon,
            action_source=action_source,
            proprio_noise_std=args.proprio_noise_std,
            batch_size=args.batch_size,
            restore=args.restore,
        )
    finally:
        tee.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
