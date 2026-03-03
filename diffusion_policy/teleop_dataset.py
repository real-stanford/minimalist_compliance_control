"""Teleoperation dataset for manipulation tasks."""

from typing import List

import joblib
import numpy as np
import numpy.typing as npt
import torch

from .utils.dataset_utils import (
    create_sample_indices,
    create_video_grid,
    get_data_stats,
    normalize_data,
    sample_sequence,
)


class TeleopImageDataset(torch.utils.data.Dataset):
    """Dataset class for teleoperation data with images."""

    def __init__(
        self,
        dataset_path_list: List[str],
        exp_folder_path: str,
        pred_horizon: int,
        obs_horizon: int,
        obs_source: list[str] | str | None = None,
        image_views: str = "left",
        image_horizon: int = 1,
        action_source: list[str] | str | None = None,
    ):
        """Initializes the data processing pipeline for a teleoperation dataset.

        This constructor loads multiple datasets, processes and normalizes the data, and prepares it for model training or evaluation. It handles image, observation, and action data, creating video visualizations and plots for analysis. The method also computes sample indices for state-action sequences, considering prediction, observation, and action horizons.

        Args:
            dataset_path_list (List[str]): List of file paths to the datasets to be loaded.
            exp_folder_path (str): Path to the folder where experiment outputs, such as videos and plots, will be saved.
            pred_horizon (int): The prediction horizon length for the model.
            obs_horizon (int): The observation horizon length for the model.
            obs_source (list[str] | str, optional): Observation keys to use. Defaults to None.
            image_views (str, optional): Image views to load: left, right, or both. Defaults to "left".
            image_horizon (int, optional): Number of image frames to keep. Defaults to 1.
            action_source (list[str] | str, optional): Action key to use. Defaults to None.
        """

        def normalize_obs_source(
            source: list[str] | str | None,
        ) -> list[str] | None:
            if source is None:
                return None
            items = (
                [source] if isinstance(source, str) else [str(item) for item in source]
            )
            cleaned = [
                token.strip()
                for item in items
                for token in item.split(",")
                if token.strip()
            ]
            return cleaned or None

        def normalize_image_views(value: str | None) -> str:
            cleaned = (value or "left").strip().lower()
            if cleaned not in {"left", "right", "both"}:
                raise ValueError(
                    f"image_views must be one of left, right, both; got {value!r}"
                )
            return cleaned

        def load_image_data(
            dataset_root: dict, image_view: str
        ) -> tuple[np.ndarray, np.ndarray]:
            if image_view == "left":
                image_data = dataset_root.get("left_image")
                if image_data is None:
                    raise KeyError(
                        "Dataset missing left_image entry. "
                        "Regenerate datasets with left_image/right_image keys."
                    )
                return image_data, image_data
            if image_view == "right":
                image_data = dataset_root.get("right_image")
                if image_data is None:
                    raise KeyError(
                        "Dataset missing right_image entry. "
                        "Regenerate datasets with left_image/right_image keys."
                    )
                return image_data, image_data
            if image_view == "both":
                left_image = dataset_root.get("left_image")
                right_image = dataset_root.get("right_image")
                missing = [
                    name
                    for name, data in (
                        ("left_image", left_image),
                        ("right_image", right_image),
                    )
                    if data is None
                ]
                if missing:
                    missing_str = ", ".join(missing)
                    raise KeyError(
                        f"Dataset missing {missing_str} entry for image_views=both. "
                        "Regenerate datasets with left_image/right_image keys."
                    )
                if left_image.shape != right_image.shape:
                    raise ValueError(
                        "left_image and right_image shapes must match for image_views=both; "
                        f"got {left_image.shape} vs {right_image.shape}."
                    )
                combined = np.concatenate([left_image, right_image], axis=-1)
                return combined, left_image
            raise ValueError(f"Unsupported image_views: {image_view}")

        requested_obs_source = normalize_obs_source(obs_source)
        requested_action_source = normalize_obs_source(action_source)
        image_views = normalize_image_views(image_views)
        if image_horizon < 1:
            raise ValueError("image_horizon must be >= 1.")
        train_image_list = []
        train_image_vis_list = []
        train_obs_list = []
        train_action_list = []
        episode_ends_list: List[npt.NDArray[np.float32]] = []
        action_source_label: str | None = None
        obs_source_label: list[str] | None = None
        for dataset_path in dataset_path_list:
            dataset_root = joblib.load(dataset_path)
            image_data, image_vis_data = load_image_data(dataset_root, image_views)
            train_image_list.append(np.moveaxis(image_data, -1, 1))
            train_image_vis_list.append(np.moveaxis(image_vis_data, -1, 1))
            dataset_action = dataset_root["action"]
            dataset_obs = dataset_root["obs"]
            dataset_action_source = normalize_obs_source(
                dataset_root.get("action_source")
            )
            dataset_obs_source = normalize_obs_source(dataset_root.get("obs_source"))
            dataset_obs_dims = dataset_root.get("obs_source_dims")
            if isinstance(dataset_action, dict):
                if requested_action_source is None:
                    if dataset_action_source is None:
                        dataset_action_source = list(dataset_action.keys())
                    action_keys = (
                        dataset_action_source
                        if isinstance(dataset_action_source, list)
                        else [dataset_action_source]
                    )
                else:
                    if len(requested_action_source) != 1:
                        raise ValueError(
                            "action_source must specify exactly one entry; reprocess datasets or pick one action source."
                        )
                    action_keys = requested_action_source
                action_key = action_keys[0] if action_keys else None
                if action_key is None or action_key not in dataset_action:
                    raise ValueError(
                        "action_source missing or not found in action entries; reprocess datasets."
                    )
                dataset_action = dataset_action[action_key]
                dataset_action_source = action_key
            else:
                if isinstance(dataset_action_source, list):
                    if len(dataset_action_source) == 1:
                        dataset_action_source = dataset_action_source[0]
                    elif dataset_action_source:
                        raise ValueError(
                            "action_source has multiple entries but action is not a dict; reprocess datasets."
                        )
                if requested_action_source is not None:
                    if dataset_action_source is None:
                        raise ValueError(
                            "action_source requested but dataset missing action_source metadata; reprocess datasets."
                        )
                    if (
                        len(requested_action_source) != 1
                        or requested_action_source[0] != dataset_action_source
                    ):
                        raise ValueError(
                            "Requested action_source does not match dataset action_source; reprocess datasets."
                        )
            if dataset_action_source is not None:
                if action_source_label is None:
                    action_source_label = dataset_action_source
                elif dataset_action_source != action_source_label:
                    raise ValueError(
                        "Mixed action sources in datasets. Use a consistent action_source."
                    )
            train_action_list.append(dataset_action)

            if isinstance(dataset_obs, dict):
                if dataset_obs_source is None:
                    dataset_obs_source = list(dataset_obs.keys())
                if dataset_obs_dims is None:
                    obs_dims_map = {
                        key: int(dataset_obs[key].shape[1])
                        for key in dataset_obs_source
                    }
                elif isinstance(dataset_obs_dims, dict):
                    obs_dims_map = {
                        key: int(value) for key, value in dataset_obs_dims.items()
                    }
                else:
                    obs_dims_map = {
                        key: int(dim)
                        for key, dim in zip(dataset_obs_source, dataset_obs_dims)
                    }
                if requested_obs_source is None:
                    selected_keys = dataset_obs_source
                else:
                    missing = [
                        key
                        for key in requested_obs_source
                        if key not in dataset_obs_source
                    ]
                    if missing:
                        raise ValueError(
                            f"obs_source keys not in dataset: {sorted(missing)}"
                        )
                    selected_keys = requested_obs_source
                obs_chunks = [dataset_obs[key] for key in selected_keys]
                dataset_obs = (
                    np.concatenate(obs_chunks, axis=1)
                    if len(obs_chunks) > 1
                    else obs_chunks[0]
                )
                dataset_obs_source = selected_keys
                dataset_obs_dims = [obs_dims_map[key] for key in dataset_obs_source]
            else:
                if dataset_obs_dims is not None:
                    dataset_obs_dims = [int(dim) for dim in dataset_obs_dims]
                if requested_obs_source is not None:
                    if dataset_obs_source is None:
                        raise ValueError(
                            "obs_source requested but dataset missing obs_source metadata; reprocess datasets."
                        )
                    missing = [
                        key
                        for key in requested_obs_source
                        if key not in dataset_obs_source
                    ]
                    if missing:
                        raise ValueError(
                            f"obs_source keys not in dataset: {sorted(missing)}"
                        )
                    if dataset_obs_source != requested_obs_source:
                        if dataset_obs_dims is None:
                            raise ValueError(
                                "obs_source subset requested but dataset missing obs_source_dims; reprocess datasets."
                            )
                        if len(dataset_obs_dims) != len(dataset_obs_source):
                            raise ValueError(
                                "obs_source_dims does not match obs_source length."
                            )
                        offsets = np.cumsum([0] + dataset_obs_dims)
                        obs_chunks = []
                        selected_dims = []
                        for key in requested_obs_source:
                            idx = dataset_obs_source.index(key)
                            start = int(offsets[idx])
                            end = int(offsets[idx + 1])
                            obs_chunks.append(dataset_obs[:, start:end])
                            selected_dims.append(dataset_obs_dims[idx])
                        dataset_obs = (
                            np.concatenate(obs_chunks, axis=1)
                            if len(obs_chunks) > 1
                            else obs_chunks[0]
                        )
                        dataset_obs_source = requested_obs_source
                        dataset_obs_dims = selected_dims
                    else:
                        dataset_obs_source = requested_obs_source
            if dataset_obs_source is not None:
                if obs_source_label is None:
                    obs_source_label = dataset_obs_source
                elif dataset_obs_source != obs_source_label:
                    raise ValueError(
                        "Mixed obs sources in datasets. Use a consistent obs_source."
                    )
            train_obs_list.append(dataset_obs)
            if len(episode_ends_list) > 0:
                episode_ends_list.append(
                    episode_ends_list[-1][-1] + dataset_root["episode_ends"]
                )
            else:
                episode_ends_list.append(dataset_root["episode_ends"])

        # concatenate all the data
        train_image_data = np.concatenate(train_image_list, axis=0).astype(
            np.float32, copy=False
        )
        if image_views == "both":
            train_image_vis_data = np.concatenate(train_image_vis_list, axis=0)
        else:
            train_image_vis_data = train_image_data
        train_obs = np.concatenate(train_obs_list, axis=0).astype(
            np.float32, copy=False
        )
        train_action = np.concatenate(train_action_list, axis=0).astype(
            np.float32, copy=False
        )
        episode_ends = np.concatenate(episode_ends_list, axis=0)

        create_video_grid(
            train_image_vis_data, episode_ends, exp_folder_path, "image_data.mp4"
        )
        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=max(obs_horizon, image_horizon) - 1,
            pad_after=pred_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in zip(["obs", "action"], [train_obs, train_action]):
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key]).astype(
                np.float32, copy=False
            )

        # images are already normalized
        normalized_train_data["image"] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.image_horizon = image_horizon
        self.action_source_label = action_source_label
        self.action_is_relative = action_source_label in {
            "x_cmd_relative",
            "x_cmd_delta",
        }

    def __len__(self):
        """Returns the number of elements in the collection.

        This method provides the length of the collection by returning the count of indices stored.

        Returns:
            int: The number of elements in the collection.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """Retrieves a normalized data sample for a given index.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            dict: A dictionary containing the normalized data sample with keys 'image' and 'obs', each truncated to the observation horizon.
        """
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["image"] = nsample["image"][: self.image_horizon, :]
        nsample["obs"] = nsample["obs"][: self.obs_horizon, :]
        if self.action_is_relative and sample_end_idx < self.pred_horizon:
            nsample["action"][sample_end_idx:] = 0.0
        return nsample
