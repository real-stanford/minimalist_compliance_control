"""Minimal ZMQ keyboard command utilities for model_based.

Commands:
- c: reverse current goal direction
- l: switch to left-hand mode
- r: switch to right-hand mode
- b: switch to both-hands mode
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


VALID_COMMANDS = {"c", "l", "r", "b"}


@dataclass
class KeyboardCommand:
    command: str
    recv_time: float


class KeyboardControlReceiver:
    """Non-blocking ZMQ receiver (PULL) for keyboard commands."""

    def __init__(self, port: int = 5592) -> None:
        self.port = int(port)
        self.enabled = False
        self._ctx = None
        self._sock = None
        self._zmq = None

        try:
            import zmq  # type: ignore
        except Exception as exc:
            print(
                "[model_based] Keyboard control disabled: "
                f"pyzmq unavailable ({exc})."
            )
            return

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://*:{self.port}")
        self.enabled = True
        print(
            "[model_based] Keyboard receiver listening on "
            f"tcp://*:{self.port} (c/l/r/b)."
        )

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None

    def poll_command(self) -> Optional[KeyboardCommand]:
        if not self.enabled or self._sock is None or self._zmq is None:
            return None

        try:
            msg = self._sock.recv_json(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return None
        except Exception:
            return None

        if isinstance(msg, dict):
            raw_cmd = msg.get("command", "")
        else:
            raw_cmd = msg

        cmd = str(raw_cmd).strip().lower()
        if cmd not in VALID_COMMANDS:
            return None

        return KeyboardCommand(command=cmd, recv_time=time.time())


class KeyboardControlSender:
    """Simple ZMQ sender (PUSH) for keyboard commands."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5592) -> None:
        self.host = str(host)
        self.port = int(port)
        self._ctx = None
        self._sock = None

        try:
            import zmq  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "pyzmq is required for keyboard control sender"
            ) from exc

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self.host}:{self.port}")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None

    def send(self, command: str) -> bool:
        if self._sock is None:
            return False

        cmd = str(command).strip().lower()
        if cmd not in VALID_COMMANDS:
            return False

        payload = {"command": cmd, "time": time.time()}
        self._sock.send_json(payload)
        return True
