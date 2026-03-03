# real_world

This folder contains hardware-facing components:

- `real_world_dynamixel.py`: real robot adapter for Dynamixel-based robots
  (for example, Toddlerbot).
- `real_world_arx.py`: real robot adapter for ARX motor setups.
- `real_world_g1.py`: real robot adapter for Unitree G1.
- `arx_controller.py`: ARX5 controller interface.
- `g1_controller.py`: Unitree G1 UDP/command interface.
- `calibrate_zero.py`: zero-offset calibration utility for supported motors.
- `dynamixel/`: Dynamixel/ARX C++ and Python bridge code.
- `IMU.py`: BNO08X IMU interface and threaded reader.
- `camera.py`: Stereo camera helper.
- `camera.yml`: Legacy optional camera control defaults.

Environment variables:

- `MCC_ROBOT`: robot camera profile key (for default `assets/<robot>_camera.yml`).
- `MCC_CAMERA_CONFIG`: override camera config YAML path.
