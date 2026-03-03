"""Inertial Measurement Unit (IMU) sensor interface module.

Provides IMU and ThreadedIMU classes for reading orientation and angular velocity
data from BNO08X sensor with optional Butterworth filtering and high-frequency sampling.
"""

# from toddlerbot.utils.misc_utils import profile
import importlib.util
import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter
from scipy.spatial.transform import Rotation as R

try:
    import board
    import busio
    from adafruit_bno08x import (
        BNO_REPORT_GAME_ROTATION_VECTOR,
        # BNO_REPORT_GRAVITY,
        BNO_REPORT_GYROSCOPE,
        # BNO_REPORT_LINEAR_ACCELERATION,
        BNO_REPORT_ROTATION_VECTOR,
    )
    from adafruit_bno08x.i2c import BNO08X_I2C
except Exception:
    board = None
    busio = None
    BNO08X_I2C = None

# from toddlerbot.utils.math_utils import butterworth


def _resolve_bno08x_init_file() -> Path | None:
    spec = importlib.util.find_spec("adafruit_bno08x")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin)
    return path if path.exists() else None


def set_report_interval(report_interval: int = 5000):
    """Sets the _DEFAULT_REPORT_INTERVAL in the BNO08X driver to the specified value.

    Args:
        report_interval (int): New interval in microseconds. Defaults to 200000 (200ms).
    """
    driver_file = _resolve_bno08x_init_file()
    if driver_file is None:
        print(
            "Warning: adafruit_bno08x package not found. Skipping report interval set."
        )
        return
    driver_path = str(driver_file)

    if not os.path.exists(driver_path):
        print(f"Error: adafruit_bno08x.py not found at {driver_path}")
        return

    try:
        with open(driver_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        modified = False
        for i, line in enumerate(lines):
            if "_DEFAULT_REPORT_INTERVAL" in line and "const" in line:
                current_value = int(line.split("(")[-1].split(")")[0])
                if current_value != report_interval:
                    lines[i] = (
                        f"_DEFAULT_REPORT_INTERVAL = const({report_interval})  # in microseconds\n"
                    )
                    modified = True
                break

        if modified:
            with open(driver_path, "w", encoding="utf-8") as file:
                file.writelines(lines)
            print(f"_DEFAULT_REPORT_INTERVAL set to {report_interval} in {driver_path}")
        else:
            print(
                f"_DEFAULT_REPORT_INTERVAL is already set to {report_interval}, no changes made."
            )

    except Exception as e:
        print(f"Error while modifying the file: {e}")


class IMU:
    """Class for interfacing with the BNO08X IMU sensor."""

    def __init__(
        self,
        address: int = 0x4A,
        offset: float = -np.pi / 2,
        mount_offset_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_game_rotation: bool = True,
    ):
        """Initializes the sensor interface.

        Args:
            rot_alpha (float): Smoothing factor for rotation. Defaults to 1.0 (no smoothing).
        """
        if board is None or busio is None or BNO08X_I2C is None:
            raise ImportError(
                "IMU dependencies are missing. Install `adafruit-circuitpython-bno08x` "
                "and hardware I2C dependencies before using IMU."
            )
        set_report_interval()

        # Initialize the I2C bus and sensor
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = BNO08X_I2C(self.i2c, address=address)
        self.use_game_rotation = use_game_rotation

        # Enable the gyroscope and rotation vector features
        # self.sensor.enable_feature(BNO_REPORT_GRAVITY)
        self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)
        # self.sensor.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
        if use_game_rotation:
            self.sensor.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
        else:
            self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)

        time.sleep(0.2)

        self.is_open = True
        self.zero_yaw = None
        base_zero_euler = np.array([0.0, float(offset), 0.0], dtype=np.float32)
        base_zero_rot = R.from_euler("xyz", base_zero_euler)
        self.mount_offset_euler = np.array(mount_offset_euler, dtype=np.float32)
        mount_offset_rot = R.from_euler("xyz", self.mount_offset_euler)
        self.zero_rot = base_zero_rot * mount_offset_rot
        self.zero_rot_inv = self.zero_rot.inv()

    # @profile()
    def get_state(self):
        """Computes and returns the current state of the system, including filtered Euler angles and angular velocity.

        This function processes raw sensor data to compute the relative rotation and angular velocity of the system. It applies an exponential moving average to filter the Euler angles and angular velocity, ensuring smoother transitions. The function returns these values in a dictionary format.

        Returns:
            Dict[str, npt.NDArray[np.float32]]: A dictionary containing:
                - "euler": The filtered Euler angles as a NumPy array.
                - "ang_vel": The filtered angular velocity as a NumPy array.
        """
        if not self.is_open:
            raise RuntimeError("IMU is not open. Please initialize the sensor first.")

        if self.use_game_rotation:
            quat_raw = np.array(
                self.sensor.game_quaternion, dtype=np.float32, copy=True
            )
        else:
            quat_raw = np.array(self.sensor.quaternion, dtype=np.float32, copy=True)

        rot_raw = R.from_quat(quat_raw)
        if self.zero_yaw is None:
            self.zero_yaw = rot_raw.as_euler("xyz")[2]

        rot = rot_raw * self.zero_rot_inv

        euler = rot.as_euler("xyz")
        euler[2] -= self.zero_yaw
        quat = R.from_euler("xyz", euler).as_quat(scalar_first=True)

        ang_vel_raw = np.array(self.sensor.gyro, dtype=np.float32, copy=True)
        ang_vel = self.zero_rot.apply(ang_vel_raw)

        return quat, ang_vel

    def close(self):
        """Close the IMU sensor connection."""
        self.is_open = False


class ThreadedIMU:
    """Threaded IMU class with high-frequency sampling and Butterworth filtering.

    Device: 200 Hz gyro + (optional) rotation vector; consume all packets; record timestamps.
    Filter: 4th-order LP with fc≈20–22 Hz at 200 Hz; then decimate to 50 Hz.
    """

    def __init__(
        self,
        input_freq: float = 200.0,
        output_freq: float = 200.0,
        cutoff_freq: float = 50.0,
        filter_order: int = 4,
        imu_mount_offset_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Initialize ThreadedIMU with Butterworth filtering.

        Args:
            dt: Target sampling period in seconds (0.02 = 50Hz output)
            maxlen: Maximum length of data queue
            cutoff_freq: Butterworth filter cutoff frequency in Hz (20-22 Hz recommended)
            filter_order: Butterworth filter order (4th order recommended)
        """
        # IMU runs at 200 Hz internally, we decimate to 50 Hz
        self.input_freq = input_freq  # Hz - internal IMU sampling rate
        self.output_freq = output_freq  # Hz - output rate after decimation
        self.input_dt = 1 / self.input_freq
        self.output_dt = 1 / self.output_freq
        self.decimation_factor = int(self.input_freq / self.output_freq)

        # Single-IMU setup.
        self.imu = IMU(address=0x4A, mount_offset_euler=imu_mount_offset_euler)

        # Threading controls
        self.running = False
        self.lock = threading.Lock()
        self.latest_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.thread = None

        # Butterworth filter setup
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        nyquist = 0.5 * self.input_freq
        normalized_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(
            filter_order, normalized_cutoff, btype="low", analog=False
        )

        # Filter state for angular velocity (3 axes).
        self.ang_vel_past_inputs = np.zeros((len(self.b) - 1, 3), dtype=np.float32)
        self.ang_vel_past_outputs = np.zeros((len(self.a) - 1, 3), dtype=np.float32)

        # Decimation counter
        self.sample_counter = 0

    def start(self):
        """Start the IMU data collection thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Main data collection loop running at specified input frequency."""
        while self.running:
            try:
                # Get raw IMU data with timeout
                t_start = time.monotonic()
                quat_raw, ang_vel_raw = self.imu.get_state()

                # Apply Butterworth filter to angular velocity.
                # (
                #     filtered_ang_vel,
                #     self.ang_vel_past_inputs,
                #     self.ang_vel_past_outputs,
                # ) = butterworth(
                #     self.b,
                #     self.a,
                #     ang_vel_raw,
                #     self.ang_vel_past_inputs,
                #     self.ang_vel_past_outputs,
                # )
                filtered_ang_vel = ang_vel_raw

                t_end = time.monotonic()

                # Decimation: only output every Nth sample to achieve 50 Hz
                self.sample_counter += 1
                if self.sample_counter >= self.decimation_factor:
                    self.sample_counter = 0

                    with self.lock:
                        # 2-item tuple: (quat, ang_vel)
                        self.latest_state = (quat_raw.copy(), filtered_ang_vel.copy())

                remaining_time = self.input_dt - (t_end - t_start)
                # print(f"[ThreadedIMU] Remaining time: {remaining_time:.3f} s")
                time.sleep(max(0, remaining_time))

            except Exception as e:
                print(f"[ThreadedIMU] Error: {e}")

    def get_latest_state(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the latest filtered IMU state.

        Returns:
            2-item tuple (quat, ang_vel).
        """
        with self.lock:
            return self.latest_state

    def close(self):
        """Stop the IMU thread and close the connection."""
        self.running = False
        if self.thread:
            self.thread.join()
        self.imu.close()
