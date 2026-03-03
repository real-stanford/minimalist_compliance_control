#!/usr/bin/env python3
"""Launch Foundation Stereo and SAM3 servers with default settings."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys

from vlm.servers.foundation_stereo_server import FoundationStereoServer
from vlm.servers.sam3_server import SAM3Server

# Default ZeroMQ configuration
DEFAULT_RESPONSE_IP = "127.0.0.1"
QUEUE_LEN = 1

FS_REQUEST_PORT = 5555
FS_RESPONSE_PORT = 5556
FS_LOOP_RATE = 50.0

SAM3_REQUEST_PORT = 5580
SAM3_RESPONSE_PORT = 5581
SAM3_LOOP_RATE = 50.0


def start_foundation_stereo(response_ip: str) -> None:
    server = FoundationStereoServer(
        response_ip=response_ip,
        request_port=FS_REQUEST_PORT,
        response_port=FS_RESPONSE_PORT,
        loop_rate_hz=FS_LOOP_RATE,
        queue_len=QUEUE_LEN,
    )
    server.serve_forever()


def start_sam3(response_ip: str) -> None:
    server = SAM3Server(
        response_ip=response_ip,
        request_port=SAM3_REQUEST_PORT,
        response_port=SAM3_RESPONSE_PORT,
        loop_rate_hz=SAM3_LOOP_RATE,
        queue_len=QUEUE_LEN,
    )
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Launch Foundation servers (Stereo, SAM3).")
    )
    parser.add_argument(
        "--ip",
        default=DEFAULT_RESPONSE_IP,
        help="Response IP for ZeroMQ sockets (default: %(default)s)",
    )
    args = parser.parse_args()
    response_ip = args.ip

    processes = [
        mp.Process(
            target=start_foundation_stereo,
            args=(response_ip,),
            name="FoundationStereoServer",
        ),
        mp.Process(
            target=start_sam3,
            args=(response_ip,),
            name="SAM3Server",
        ),
    ]

    for process in processes:
        process.start()

    def shutdown(signum, frame):  # noqa: D401, U100
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for process in processes:
        process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
