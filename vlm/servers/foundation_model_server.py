#!/usr/bin/env python3
"""Base classes and utilities for foundation model servers."""

from __future__ import annotations

import datetime
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TypeVar

import numpy as np
import zmq

from vlm.utils.comm_utils import ZMQNode

T = TypeVar("T")


def ensure_array(value: Any) -> np.ndarray:
    """Convert lists or numpy arrays into an ndarray."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.array(value)
    raise TypeError("Expected numpy array or sequence")


def ensure_image_array(value: Any) -> np.ndarray:
    """Ensure the value can be interpreted as an image array."""
    array = ensure_array(value)
    if array.ndim not in (2, 3):
        raise ValueError("Image arrays must be 2D or 3D")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


class FoundationModelServer(ABC):
    """Common ZeroMQ server functionality for foundation models."""

    model_key: str = ""

    def __init__(
        self,
        *,
        response_ip: str,
        request_port: int,
        response_port: int,
        loop_rate_hz: float = 50.0,
        queue_len: int = 1,
    ) -> None:
        if loop_rate_hz <= 0:
            raise ValueError("loop_rate_hz must be positive")

        if not self.model_key:
            raise ValueError("model_key must be defined by subclasses")

        self.loop_period = 1.0 / loop_rate_hz
        self.poll_timeout_ms = max(int(self.loop_period * 1000), 1)
        self.response_ip = response_ip
        self.request_port = request_port
        self.response_port = response_port

        self.request_node = ZMQNode(
            type="receiver",
            queue_len=queue_len,
            port=request_port,
        )
        self.response_node = ZMQNode(
            type="sender",
            ip=response_ip,
            queue_len=queue_len,
            port=response_port,
        )
        self.poller = zmq.Poller()
        self.poller.register(self.request_node.socket, zmq.POLLIN)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release ZeroMQ sockets held by this server."""
        if (
            getattr(self, "poller", None) is not None
            and getattr(self.request_node, "socket", None) is not None
        ):
            try:
                self.poller.unregister(self.request_node.socket)
            except (KeyError, ValueError):
                pass
        if getattr(self, "request_node", None) is not None:
            self.request_node.close()
        if getattr(self, "response_node", None) is not None:
            self.response_node.close()

    def serve_forever(self) -> None:
        """Run the service loop."""
        self.log(f"Listening on port {self.request_port} for model '{self.model_key}'")
        try:
            while True:
                events = dict(self.poller.poll(self.poll_timeout_ms))
                if not events:
                    continue
                if (
                    self.request_node.socket in events
                    and events[self.request_node.socket] & zmq.POLLIN
                ):
                    message, recv_duration = self.measure_duration(
                        self.request_node.get_msg
                    )
                    if message is None:
                        continue
                    self.log(
                        f"Receive request completed in {recv_duration * 1000:.1f} ms; dispatching handler"
                    )
                    response, process_duration = self.measure_duration(
                        self.process_message, message
                    )
                    self.log(
                        f"Request processing finished in {process_duration * 1000:.1f} ms"
                    )
                    _, send_duration = self.measure_duration(
                        self.send_response, response
                    )
                    self.log(
                        f"Response transmission finished in {send_duration * 1000:.1f} ms"
                    )
        except KeyboardInterrupt:
            self.log("Stopping server loop")

    def process_message(self, message: Any) -> Dict[str, Any]:
        """Validate incoming payload and delegate to subclass handler."""
        model_name: Optional[str] = None
        try:
            if not isinstance(message, dict):
                raise TypeError(
                    "Expected dict payload with 'model', 'data', and 'config'"
                )

            model_name = message.get("model")
            if model_name != self.model_key:
                raise ValueError(
                    f"{self.__class__.__name__} expects model '{self.model_key}' "
                    f"but received '{model_name}'"
                )

            data = message.get("data") or {}
            config = message.get("config") or {}

            result, inference_duration = self.measure_duration(
                self.handle_request, data, config
            )
            self.log(f"Inference completed in {inference_duration * 1000:.1f} ms")
            self.log("Request completed successfully")
            return {
                "status": "ok",
                "model": model_name,
                "result": result,
            }
        except Exception as exc:  # pylint: disable=broad-except
            self.log(f"Request failed: {exc}")
            return {
                "status": "error",
                "model": model_name,
                "error": str(exc),
            }

    def send_response(self, response: Dict[str, Any]) -> None:
        """Send a serialized response through the configured ZMQ sender."""
        try:
            payload = pickle.dumps(response)
            self.response_node.socket.send(payload, flags=zmq.NOBLOCK)
            self.log(
                "Response sent "
                f"(status={response.get('status')}, model={response.get('model')})"
            )
        except zmq.Again:
            self.log("Response queue is full; dropping reply")
        except Exception as exc:  # pragma: no cover - defensive
            self.log(f"Failed to send response: {exc}")

    def log(self, message: str) -> None:
        pst_timezone = datetime.timezone(datetime.timedelta(hours=-8))
        timestamp = datetime.datetime.now(pst_timezone)
        print(f"[{timestamp:%Y-%m-%d %H:%M:%S}] {message}")

    def measure_duration(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> tuple[T, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        return result, duration

    @abstractmethod
    def handle_request(self, data: Dict[str, Any], config: Dict[str, Any]) -> Any:
        """Process the model-specific request and return a result."""
        raise NotImplementedError
