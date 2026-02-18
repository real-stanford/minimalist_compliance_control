# real_world

This folder contains hardware-facing components:

- `dynamixel/`: Dynamixel/ARX C++ and Python bridge code.
- `IMU.py`: BNO08X IMU interface and threaded reader.
- `camera.py`: Stereo camera helper.
- `camera.yml`: Legacy optional camera control defaults.

Environment variables:

- `MCC_DYNAMIXEL_CPP_PATH`: explicit path to `dynamixel_cpp*.so`.
- `MCC_TODDLERBOT_INTERNAL_PATH`: fallback path for legacy extension lookup.
- `MCC_ROBOT`: robot camera profile key (for default `assets/<robot>_camera.yml`).
- `MCC_CAMERA_CONFIG`: override camera config YAML path.
- `MCC_CAMERA_CALIB`: override camera calibration `.pkl` path.
