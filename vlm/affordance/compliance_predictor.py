#!/usr/bin/env python3
"""
Predictor Core Module
Main affordance prediction logic with grid-based and robotics model support
"""

import base64
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from .model_provider import ModelProvider

# from toddlerbot.utils.misc_utils import profile

MODEL_PROVIDERS = {
    "openai": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "auth_type": "bearer",
        "api_key_env": "OPENAI_API_KEY",
        "request_format": "openai",
    },
    "gemini": {
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "auth_type": "query_param",
        "api_key_env": "GOOGLE_API_KEY",
        "request_format": "gemini",
    },
}

API_SETTINGS = {"max_tokens": 32768, "temperature": 0.1, "timeout": 300}


class CompliancePredictor:
    def __init__(self, provider: str, model: str):
        if provider not in MODEL_PROVIDERS:
            raise ValueError(f"Provider '{provider}' not found in provider list")

        provider_config = MODEL_PROVIDERS[provider]

        self.model = model
        api_key_env = provider_config["api_key_env"]
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"API key not found. Set environment variable {api_key_env}"
            )

        self.provider = ModelProvider(provider_config, self.api_key, self.model)
        self.api_settings = API_SETTINGS

        print(f"Initialized with provider: {provider}, model: {self.model}")

    def encode_image(self, image: np.ndarray) -> str:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        success, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )
        if not success:
            raise ValueError("Failed to encode image for compliance predictor")

        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    # @profile()
    def invoke_model(self, prompt: str, images_data: str) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()
        headers = self.provider.get_headers()
        payload = self.provider.format_request(
            prompt, images_data, self.model, self.api_settings
        )

        img_b64_bytes = (
            images_data.encode("utf-8")
            if isinstance(images_data, str)
            else bytes(images_data)
        )
        payload_bytes = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        t1 = time.perf_counter()
        print(
            "[CompliancePredictor] invoke_model payload ready in "
            f"{t1 - t0:.3f}s (payload_bytes={len(payload_bytes)}, "
            f"img_b64_bytes={len(img_b64_bytes)})"
        )

        timeout = self.api_settings.get("timeout", 120)
        api_url = self.provider.get_api_url()
        full_url = self.provider.get_url_with_params(api_url)
        t2 = time.perf_counter()
        response = requests.post(
            full_url, headers=headers, json=payload, timeout=timeout
        )
        t3 = time.perf_counter()
        post_wall = t3 - t2
        response_elapsed = (
            response.elapsed.total_seconds()
            if getattr(response, "elapsed", None) is not None
            else float("nan")
        )
        print(
            "[CompliancePredictor] invoke_model POST completed in "
            f"{post_wall:.3f}s (response.elapsed={response_elapsed:.3f}s)"
        )

        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            response.raise_for_status()

        result = response.json()

        if self.provider.request_format == "openai":
            if "choices" not in result or not result["choices"]:
                print("Error: No choices in response.")
                print(f"Full API response: {result}")
                if "error" in result:
                    print(f"API Error details: {result['error']}")
                return None
        elif self.provider.request_format == "gemini":
            if "candidates" not in result or not result["candidates"]:
                print("Error: No candidates in Gemini response.")
                print(f"Full API response: {result}")
                if "error" in result:
                    print(f"API Error details: {result['error']}")
                return None

        content = self.provider.parse_response(result)

        if not content or content.strip() == "":
            print("Error: Empty response from API")
            print(f"Full API response: {result}")
            return None

        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            json_str = json_match.group() if json_match else content

        json_str = json_str.strip()
        if json_str and not json_str.startswith("{"):
            json_str = "{\n" + json_str
        if json_str and not json_str.endswith("}"):
            json_str = json_str + "\n}"

        try:
            contact_data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            print(f"Error decoding JSON: {exc}\nResponse content: {content}")
            return None

        return contact_data

    # @profile()
    def predict_compliance(
        self,
        image: np.ndarray,
        task_description: str,
        object_label: str,
        *,
        candidate_point_groups: Dict[str, List[Dict[str, int]]],
    ):
        base64_image = self.encode_image(image)

        prompt = self.create_prompt(
            task_description,
            object_label,
            candidate_point_groups=candidate_point_groups,
        )
        contact_data = self.invoke_model(prompt, base64_image)
        if not contact_data:
            return None

        # Expect a mapping from site_name -> contact description
        results: Dict[str, Any] = {}
        for site_name in candidate_point_groups:
            entry = contact_data.get(site_name)
            if not entry:
                continue
            parsed = self.parse_contact_data(entry)
            if parsed is None:
                continue
            results[site_name] = parsed

        return prompt, results

    def parse_contact_data(self, contact_data: Dict[str, Any]):
        steps = contact_data.get("contact_sequence")
        if not isinstance(steps, list) or not steps:
            steps = contact_data.get("contact_range")
        if not isinstance(steps, list) or not steps:
            print("Error: Contact list missing or empty")
            return None

        points: List[List[int]] = []
        for step in steps:
            point = step.get("contact_point")
            if not isinstance(point, list) or len(point) != 2:
                print(f"Error: contact_point missing or malformed in step: {step}")
                return None
            points.append([int(point[0]), int(point[1])])

        points_arr = np.asarray(points, dtype=np.int32)
        return points_arr

    def create_prompt(
        self,
        task_description: str,
        object_label: str,
        candidate_point_groups: Dict[str, List[Dict[str, int]]],
    ) -> str:
        """Select the appropriate prompt template based on the task description."""

        normalized_task = (task_description or "").lower()
        if "wipe" in normalized_task:
            return self.create_wipe_prompt(
                task_description,
                object_label,
                candidate_point_groups=candidate_point_groups,
            )

        if "draw" in normalized_task:
            return self.create_draw_prompt(
                task_description,
                object_label,
                candidate_point_groups=candidate_point_groups,
            )

        raise NotImplementedError(
            f"Unsupported task description for compliance prompt: {task_description}"
        )

    def format_candidate_group_lines(
        self, candidate_point_groups: Dict[str, List[Dict[str, int]]]
    ) -> str:
        if not candidate_point_groups:
            return "None provided. Select meaningful pixels directly on the target."
        blocks = []
        for site, points in candidate_point_groups.items():
            if not points:
                continue
            lines = "\n".join(
                f"ID {p['id']}: pixel=({p['x']}, {p['y']})" for p in points
            )
            blocks.append(f"[{site}]\n{lines}")
        return "\n\n".join(blocks) if blocks else "None provided."

    def format_json_lines(self, site_names: List[str]) -> str:
        names = site_names or ["<site_name>"]
        entries = []
        for site in names:
            entries.append(
                f'  "{site}": {{\n'
                '    "contact_sequence": [\n'
                '      {"contact_point": [x, y]}\n'
                "    ]\n"
                "  }"
            )
        return "{\n" + ",\n".join(entries) + "\n}"

    def create_wipe_prompt(
        self,
        task_description: str,
        object_label: str,
        candidate_point_groups: Dict[str, List[Dict[str, int]]],
    ) -> str:
        candidate_lines = self.format_candidate_group_lines(candidate_point_groups)
        site_names = list(candidate_point_groups.keys())
        json_lines = self.format_json_lines(site_names)

        prompt_template = f"""
TASK: {task_description}
TARGET OBJECT: {object_label}
CANDIDATE CONTACT POINTS (pixel coordinates), grouped by site:
{candidate_lines}

ACTION REQUIREMENTS:
- Output an ordered list of waypoints (pixel coordinates) for each site to traverse the target region. The waypoint list should define a single continuous path that sweeps through the marked region without leaving any major sub-region uncovered. Focus on path topology → coverage order, direction changes, sub-region transitions. We will densify/expand the path later using a cubic Hermite interpolation routine. Your job is only to choose the meaningful keyframes.
- Assume a small contact patch (not a large eraser): start each stroke slightly outside/upstream of the target along the motion direction (e.g., start slightly left if the next motion wipes right), then sweep through the region.

OUTPUT JSON FORMAT:
{json_lines}

JSON RULES:
✓ Return valid JSON with double quotes.
✓ Maintain the order of points exactly for dynamic sequences.
✓ Do not include explanations.

BEGIN OUTPUT:
"""

        return prompt_template

    def create_draw_prompt(
        self,
        task_description: str,
        object_label: str,
        candidate_point_groups: Dict[str, List[Dict[str, int]]],
    ) -> str:
        """Prompt for drawing trajectories (mask pixels are assumed unavailable)."""

        candidate_lines = self.format_candidate_group_lines(candidate_point_groups)
        site_names = list(candidate_point_groups.keys())
        json_lines = self.format_json_lines(site_names)

        prompt_template = f"""
TASK: {task_description}
TARGET OBJECT: {object_label}
CANDIDATE CONTACT POINTS (pixel coordinates), grouped by site:
{candidate_lines}

ACTION REQUIREMENTS:
- Output an ordered list of waypoints (pixel coordinates) for each site to traverse the target region. The waypoint list should define a single continuous path that sweeps through the marked region without leaving any major sub-region uncovered. Treat this as a single-line drawing: the pen must not leave the board. Avoid redundant connection lines; choose the waypoint order to minimize retracing. You may output extra waypoints to make the path smoother. Focus on path topology → coverage order, direction changes, sub-region transitions. We will densify/expand the path later using a cubic Hermite interpolation routine. Your job is only to choose the meaningful keyframes.

OUTPUT JSON FORMAT:
{json_lines}

JSON RULES:
✓ Return valid JSON with double quotes.
✓ Maintain the order of points exactly for dynamic sequences.
✓ Do not include explanations.

BEGIN OUTPUT:
"""

        return prompt_template
