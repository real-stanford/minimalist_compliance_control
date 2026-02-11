#!/usr/bin/env python3
"""Keyboard control sender for model_based_minimal."""

from __future__ import annotations

import argparse

from zmq_control import KeyboardControlSender


def main(host: str, port: int) -> None:
    sender = KeyboardControlSender(host=host, port=port)
    print(f"[KeyboardControl] Connected to tcp://{host}:{port}")
    print("Commands: c=reverse, l=left, r=right, b=both, q=quit")

    try:
        while True:
            cmd = input("Command (c/l/r/b/q): ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break
            if cmd not in ("c", "l", "r", "b"):
                print("Use c/l/r/b or q.")
                continue
            ok = sender.send(cmd)
            if ok:
                print(f"[KeyboardControl] Sent: {cmd}")
            else:
                print("[KeyboardControl] Failed to send command")
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
        print("[KeyboardControl] Closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Keyboard control sender for model_based_minimal"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5592)
    args = parser.parse_args()
    main(args.host, args.port)
