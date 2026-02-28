README 与实现对照简报（2026-02-28）

已核对 README 数量：6
- README.md
- vlm/README.md
- diffusion_policy/README.md
- hybrid_servo/README.md
- real_world/README.md
- descriptions/unitree_g1/README.md

总体结论：大部分说明与代码一致，但有 4 处明确出入。

1) real_world/README.md 的 3 个环境变量在代码中未实现
- 文档声明：MCC_DYNAMIXEL_CPP_PATH、MCC_TODDLERBOT_INTERNAL_PATH、MCC_CAMERA_CALIB
- 实际：全仓库仅 README 提及，未被任何代码读取。

2) real_world/README.md 提到 `real_world/camera.yml`，但仓库中无该文件
- 实际相机配置路径由 `real_world/camera.py` 指向 `assets/<robot>_camera.yml`（或 `MCC_CAMERA_CONFIG` 覆盖）。

3) descriptions/unitree_g1/README.md 引用 `g1.png`，但文件缺失
- 文档中有 `<img src="g1.png">`，目录下不存在 `descriptions/unitree_g1/g1.png`。

4) vlm/README.md 与 diffusion_policy/README.md 的“Replay Input Conventions”未找到对应多别名解析实现
- 文档给出多组别名（如 image/images/rgb/camera、x_obs/pose/ee_pose 等）。
- 实际代码未见统一别名解析入口；VLM 回放主要读取 `trajectory.lz4` 中固定键（如 `task`、`contact_pos_camera`、`contact_normals_camera`），DP 输入来源主要由 checkpoint 的 `obs_source/action_source` 决定。

其余关键项（脚本文件存在性、CLI 入口 `mcc-run-policy`、policy 可选依赖、model-based 快捷键、VLM 模式 waiting/wiping/drawing）与实现基本一致。
