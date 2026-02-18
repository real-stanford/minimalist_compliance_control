#!/usr/bin/env python3
"""SAM3 segmentation server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from vlm.servers.foundation_model_server import (
    FoundationModelServer,
    ensure_image_array,
)


class SAM3Server(FoundationModelServer):
    """Runs SAM3 image segmentation in response to ZeroMQ requests."""

    model_key = "sam3"

    def __init__(
        self,
        *,
        response_ip: str,
        request_port: int,
        response_port: int,
        loop_rate_hz: float = 50.0,
        queue_len: int = 1,
        default_prompt: str = "object.",
        model: Any = None,
        processor: Optional[Sam3Processor] = None,
        assume_rgb_input: bool = False,
    ) -> None:
        super().__init__(
            response_ip=response_ip,
            request_port=request_port,
            response_port=response_port,
            loop_rate_hz=loop_rate_hz,
            queue_len=queue_len,
        )

        self.model = model or build_sam3_image_model()
        self.processor = processor or Sam3Processor(self.model)
        self.default_prompt = default_prompt
        self.assume_rgb_input = assume_rgb_input

    def handle_request(
        self, data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run SAM3 segmentation given an image and one or more text prompts."""
        self.log("SAM3: validating request payload")
        prompts = self.resolve_prompts(data, config)
        include_masks = bool(config.get("return_masks", True))
        assume_rgb = bool(config.get("assume_rgb_input", self.assume_rgb_input))

        image = self.load_image(data, assume_rgb=assume_rgb)

        self.log(f"SAM3: running segmentation (prompts={prompts})")
        inference_state = self.processor.set_image(image)

        boxes: List[Any] = []
        scores: List[Any] = []
        masks: List[Any] = []
        prompt_ids: List[int] = []
        mask_prompt_ids: List[int] = []

        for idx, prompt in enumerate(prompts):
            output = self.processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )
            boxes_arr = self.to_numpy(output.get("boxes"))
            scores_arr = self.to_numpy(output.get("scores"))
            prompt_id_count = 0

            if boxes_arr is not None:
                if boxes_arr.ndim == 1:
                    boxes.extend([boxes_arr.tolist()])
                    prompt_id_count = 1
                else:
                    boxes_list = boxes_arr.tolist()
                    prompt_id_count = len(boxes_list)
                    boxes.extend(boxes_list)
            if scores_arr is not None:
                scores.extend(scores_arr.tolist())
            if prompt_id_count:
                prompt_ids.extend([idx] * prompt_id_count)
            masks_arr = self.to_numpy(output.get("masks"))
            if masks_arr is not None:
                masks.append(masks_arr)
                count = (
                    masks_arr.shape[0]
                    if hasattr(masks_arr, "ndim") and masks_arr.ndim > 2
                    else 1
                )
                mask_prompt_ids.extend([idx] * count)

        self.log("SAM3: segmentation complete")

        combined = {
            "boxes": boxes,
            "scores": scores,
            "masks": masks if masks else None,
            "prompt_ids": prompt_ids if prompt_ids else None,
            "mask_prompt_ids": mask_prompt_ids if mask_prompt_ids else None,
        }

        return self.serialize_output(
            combined, include_masks=include_masks, prompts=prompts
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def resolve_prompts(
        self, data: Dict[str, Any], config: Dict[str, Any]
    ) -> List[str]:
        prompt_value = (
            config.get("text_prompt")
            or config.get("prompt")
            or data.get("text_prompt")
            or data.get("prompt")
        )

        prompts: List[str] = []
        if isinstance(prompt_value, (list, tuple)):
            prompts = [str(p).strip() for p in prompt_value if str(p).strip()]
        elif prompt_value is not None:
            segments = str(prompt_value).split(".")
            prompts = [segment.strip() for segment in segments if segment.strip()]

        if not prompts:
            prompts = [self.default_prompt]

        return prompts

    def load_image(
        self, data: Dict[str, Any], *, assume_rgb: bool = False
    ) -> Image.Image:
        if "image" in data and data["image"] is not None:
            array = ensure_image_array(data["image"])
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            elif array.ndim != 3:
                raise ValueError("image array must have shape (H, W, C)")
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            if array.shape[2] == 4:
                array = array[:, :, :3]
            if not assume_rgb:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
            return Image.fromarray(array)

        if "image_path" in data and data["image_path"] is not None:
            return Image.open(data["image_path"]).convert("RGB")

        raise ValueError("SAM3 requires 'image' array or 'image_path' string in data")

    def serialize_output(
        self, output: Dict[str, Any], *, include_masks: bool, prompts: List[str]
    ) -> Dict[str, Any]:
        if not isinstance(output, dict):
            raise TypeError("SAM3 processor returned unexpected output format")

        masks = output.get("masks")
        boxes = self.to_list(output.get("boxes"))
        scores = self.to_list(output.get("scores"))
        mask_rles = self.encode_masks(masks)

        result: Dict[str, Any] = {
            "prompts": prompts,
            "boxes": boxes or [],
            "scores": scores or [],
            "mask_rles": mask_rles,
        }

        prompt_ids = self.to_list(output.get("prompt_ids"))
        if prompt_ids is not None:
            result["prompt_ids"] = prompt_ids
        mask_prompt_ids = self.to_list(output.get("mask_prompt_ids"))
        if mask_prompt_ids is not None:
            result["mask_prompt_ids"] = mask_prompt_ids

        if include_masks and masks is not None:
            result["masks"] = self.to_list(masks)

        mask_shape = self.get_shape(masks)
        if mask_shape is not None:
            result["mask_shape"] = mask_shape

        return result

    def to_numpy(self, value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            if torch is not None and any(isinstance(v, torch.Tensor) for v in value):
                converted: list[Any] = []
                for v in value:
                    if v is None:
                        continue
                    if torch.is_tensor(v):
                        converted.append(v.detach().cpu().numpy())
                    elif isinstance(v, np.ndarray):
                        converted.append(v)
                    else:
                        converted.append(v)
                try:
                    return np.stack(converted)
                except Exception:
                    return np.array(converted, dtype=object)
            try:
                return np.array(value)
            except Exception:
                return np.array(value, dtype=object)
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)

    def to_list(self, value: Any) -> Optional[Any]:
        array = self.to_numpy(value)
        if array is None:
            return None
        return array.tolist()

    def get_shape(self, value: Any) -> Optional[tuple[int, ...]]:
        array = self.to_numpy(value)
        if array is None:
            return None
        return tuple(int(dim) for dim in array.shape)

    def encode_masks(self, masks: Any) -> list[Dict[str, Any]]:
        base_array = self.to_numpy(masks)
        if base_array is None:
            return []

        masks_2d: list[np.ndarray] = []

        def collect(arr: Any) -> None:
            arr_np = np.asarray(arr)
            arr_np = np.squeeze(arr_np)
            if arr_np.ndim == 2:
                masks_2d.append(arr_np)
            elif arr_np.ndim == 3:
                for slice_arr in arr_np:
                    collect(slice_arr)
            else:
                self.log(f"SAM3: skipping mask with unsupported shape {arr_np.shape}")

        if base_array.dtype == object:
            for entry in base_array.tolist():
                collect(entry)
        else:
            collect(base_array)

        encoded: list[Dict[str, Any]] = []
        for mask_arr in masks_2d:
            mask_uint8 = np.ascontiguousarray(mask_arr.astype(np.uint8))
            rle = mask_util.encode(np.array(mask_uint8[:, :, None], order="F"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            encoded.append(rle)
        return encoded
