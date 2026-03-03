#!/usr/bin/env python3
"""
Model Provider Module
Handles API communication with different LLM providers (OpenAI, Gemini, Anthropic)
"""

from typing import Any, Dict


class ModelProvider:
    def __init__(self, config: Dict[str, Any], api_key: str, model: str = None):
        self.config = config
        self.api_key = api_key
        self.base_api_url = config["api_url"]
        self.auth_type = config["auth_type"]
        self.request_format = config["request_format"]
        self.model = model

    def get_api_url(self) -> str:
        """Get the API URL, replacing placeholders if needed."""
        if "{model}" in self.base_api_url and self.model:
            return self.base_api_url.replace("{model}", self.model)
        return self.base_api_url

    def get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}

        if self.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.auth_type == "x-api-key":
            headers["x-api-key"] = self.api_key
        # For query_param auth (like Gemini), we don't add to headers

        return headers

    def get_url_with_params(self, base_url: str) -> str:
        """Add API key as query parameter if needed."""
        if self.auth_type == "query_param":
            separator = "&" if "?" in base_url else "?"
            return f"{base_url}{separator}key={self.api_key}"
        return base_url

    def format_request(
        self, prompt: str, image_data, model: str, api_settings: Dict
    ) -> Dict:
        if self.request_format == "openai":
            return self.format_openai_request(prompt, image_data, model, api_settings)
        elif self.request_format == "gemini":
            return self.format_gemini_request(prompt, image_data, model, api_settings)
        else:
            raise ValueError(f"Unsupported request format: {self.request_format}")

    def format_openai_request(
        self, prompt: str, image_data: str, model: str, api_settings: Dict
    ) -> Dict:
        # Handle both single image (string) and multiple images (dict)
        contents = [{"type": "text", "text": prompt}]

        # Single image case (original behavior)
        contents.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            }
        )

        request_data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": contents,
                }
            ],
        }

        model_lower = model.lower()

        if any(x in model_lower for x in ["gpt-4o", "gpt-5", "o1"]):
            request_data["max_completion_tokens"] = api_settings.get("max_tokens", 1000)
        else:
            request_data["max_tokens"] = api_settings.get("max_tokens", 1000)

        if "gpt-5" not in model_lower and "o1" not in model_lower:
            request_data["temperature"] = api_settings.get("temperature", 0.1)

        return request_data

    def format_gemini_request(
        self, prompt: str, image_data: str, model: str, api_settings: Dict
    ) -> Dict:
        # Single image case
        contents = [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
        ]

        request_data = {
            "contents": [{"parts": contents}],
            "generationConfig": {
                "temperature": api_settings.get("temperature", 0.1),
                "maxOutputTokens": api_settings.get("max_tokens", 1000),
            },
        }

        # Disable thinking unless it is 2.5 pro
        thinking_budget = 0
        if "pro" in model.lower():
            thinking_budget = api_settings.get("thinking_budget_pro", 512)

        request_data["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": thinking_budget
        }

        return request_data

    def parse_response(self, response: Dict) -> str:
        if self.request_format == "openai":
            message = response["choices"][0]["message"]

            if message.get("refusal"):
                raise ValueError(f"Model refused to respond: {message['refusal']}")
            if message["content"] is None:
                raise ValueError("Model returned empty content")

            return message["content"]

        elif self.request_format == "anthropic":
            content = response["content"]
            return content[0]["text"] if content else ""
        elif self.request_format == "gemini":
            if "candidates" in response and len(response["candidates"]) > 0:
                candidate = response["candidates"][0]
                if "content" in candidate:
                    parts = candidate["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
            raise ValueError("No valid text content in Gemini response")
        else:
            raise ValueError(f"Unsupported response format: {self.request_format}")
