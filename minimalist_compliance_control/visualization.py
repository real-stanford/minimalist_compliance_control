"""Matplotlib-based compliance/wrench logging and end-of-run PNG export."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R


class CompliancePlotter:
    """Collects data online and dumps PNGs at shutdown.

    This class intentionally avoids per-step plotting so the control loop does not
    spend time in Matplotlib. PNGs are generated once in ``close()``.
    """

    def __init__(
        self,
        site_names: Sequence[str],
        enabled: bool = True,
    ) -> None:
        self.site_names = [str(name) for name in site_names]
        self.enabled = bool(enabled)
        self.error_message: Optional[str] = None
        self._has_applied_force = False
        self._hist: Dict[str, Dict[str, list[npt.NDArray[np.float64] | float]]] = {
            name: {
                "time": [],
                "cmd": [],
                "ref": [],
                "ik": [],
                "obs": [],
                "wrench": [],
                "applied_force": [],
            }
            for name in self.site_names
        }

    @staticmethod
    def _mat_to_rotvec(mat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        rot = np.asarray(mat, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(rot))
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        theta = float(np.arccos(cos_theta))
        if theta < 1e-10:
            return np.zeros(3, dtype=np.float64)
        sin_theta = float(np.sin(theta))
        if abs(sin_theta) < 1e-8:
            axis = np.sqrt(np.maximum((np.diag(rot) + 1.0) * 0.5, 0.0))
            if float(np.linalg.norm(axis)) < 1e-8:
                return np.zeros(3, dtype=np.float64)
            return axis * theta
        axis = np.array(
            [
                rot[2, 1] - rot[1, 2],
                rot[0, 2] - rot[2, 0],
                rot[1, 0] - rot[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * sin_theta)
        return axis * theta

    def update_from_wrench_sim(
        self,
        *,
        time_s: float,
        command_pose: npt.NDArray[np.float32],
        x_ref: Optional[npt.NDArray[np.float32]],
        x_ik: Optional[npt.NDArray[np.float32]],
        x_obs: npt.NDArray[np.float32],
        wrenches: Dict[str, npt.NDArray[np.float32]],
        applied_site_forces: Optional[npt.NDArray[np.float32]],
    ) -> None:
        if not self.enabled:
            return
        num_sites = len(self.site_names)
        cmd = np.asarray(command_pose, dtype=np.float64)
        if cmd.shape != (num_sites, 6):
            return
        ref = np.asarray(x_ref, dtype=np.float64) if x_ref is not None else cmd
        if ref.shape != (num_sites, 6):
            ref = cmd
        ik = np.asarray(x_ik, dtype=np.float64) if x_ik is not None else ref
        if ik.shape != (num_sites, 6):
            ik = ref
        applied_force = None
        if applied_site_forces is not None:
            applied_force_arr = np.asarray(applied_site_forces, dtype=np.float64)
            if applied_force_arr.shape == (num_sites, 3):
                applied_force = applied_force_arr
                self._has_applied_force = True
        obs = np.asarray(x_obs, dtype=np.float64)
        if obs.shape != (num_sites, 6):
            return
        for idx, site in enumerate(self.site_names):
            wrench = np.asarray(
                wrenches.get(site, np.zeros(6)), dtype=np.float64
            ).reshape(6)
            hist = self._hist[site]
            hist["time"].append(float(time_s))
            hist["cmd"].append(cmd[idx].copy())
            hist["ref"].append(ref[idx].copy())
            hist["ik"].append(ik[idx].copy())
            hist["obs"].append(obs[idx].copy())
            hist["wrench"].append(wrench.copy())
            if applied_force is not None:
                hist["applied_force"].append(applied_force[idx].copy())
            else:
                hist["applied_force"].append(np.zeros(3, dtype=np.float64))

    def _dump_pngs(self, exp_folder_path: str = "") -> None:
        if not exp_folder_path:
            return
        os.makedirs(exp_folder_path, exist_ok=True)
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.error_message = f"Matplotlib import failed in close(): {exc}"
            return

        num_sites = max(1, len(self.site_names))
        pose_w = max(12, 3.6 * num_sites)
        wrench_w = max(10, 3.6 * num_sites)

        pose_fig, pose_axes = plt.subplots(
            6, num_sites, sharex=True, figsize=(pose_w, 10.5), dpi=120, squeeze=False
        )
        wrench_fig, wrench_axes = plt.subplots(
            2, num_sites, sharex=True, figsize=(wrench_w, 5.0), dpi=120, squeeze=False
        )

        y_labels = ("x (m)", "y (m)", "z (m)", "rx (rad)", "ry (rad)", "rz (rad)")
        for col, site in enumerate(self.site_names):
            hist = self._hist[site]
            if len(hist["time"]) < 2:
                continue
            t = np.asarray(hist["time"], dtype=np.float64)
            t = t - float(t[0])
            cmd = np.asarray(hist["cmd"], dtype=np.float64)
            ref = np.asarray(hist["ref"], dtype=np.float64)
            ik = np.asarray(hist["ik"], dtype=np.float64)
            obs = np.asarray(hist["obs"], dtype=np.float64)
            wrench = np.asarray(hist["wrench"], dtype=np.float64)
            applied_force = np.asarray(hist["applied_force"], dtype=np.float64)

            for row in range(6):
                ax = pose_axes[row, col]
                ax.plot(t, cmd[:, row], color="tab:red", lw=1.3, label="cmd")
                ax.plot(
                    t, ref[:, row], color="tab:orange", lw=1.3, ls="--", label="ref"
                )
                ax.plot(t, ik[:, row], color="tab:purple", lw=1.3, ls="-.", label="ik")
                ax.plot(t, obs[:, row], color="tab:blue", lw=1.3, label="obs")
                if row == 0:
                    ax.set_title(site)
                if col == 0:
                    ax.set_ylabel(y_labels[row])
                if row == 5:
                    ax.set_xlabel("Time (s)")
                if col == 0 and row == 0:
                    ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.25)

            ax_f = wrench_axes[0, col]
            ax_t = wrench_axes[1, col]
            labels_f = ("Fx", "Fy", "Fz")
            labels_t = ("Tx", "Ty", "Tz")
            colors = ("tab:red", "tab:green", "tab:blue")
            for i in range(3):
                ax_f.plot(t, wrench[:, i], color=colors[i], lw=1.3, label=labels_f[i])
                if self._has_applied_force:
                    ax_f.plot(
                        t,
                        applied_force[:, i],
                        color=colors[i],
                        lw=1.2,
                        ls="--",
                        label=f"{labels_f[i]}_applied",
                    )
                ax_t.plot(
                    t, wrench[:, 3 + i], color=colors[i], lw=1.3, label=labels_t[i]
                )
            ax_f.set_title(f"{site} wrench")
            if col == 0:
                ax_f.set_ylabel("Force (N)")
                ax_t.set_ylabel("Torque (Nm)")
                ax_f.legend(loc="upper right", fontsize=8)
                ax_t.legend(loc="upper right", fontsize=8)
            ax_t.set_xlabel("Time (s)")
            ax_f.grid(True, alpha=0.25)
            ax_t.grid(True, alpha=0.25)

        pose_fig.tight_layout()
        wrench_fig.tight_layout()
        pose_fig.savefig(os.path.join(exp_folder_path, "compliance_ref.png"), dpi=150)
        wrench_fig.savefig(
            os.path.join(exp_folder_path, "estimated_wrench.png"), dpi=150
        )
        plt.close(pose_fig)
        plt.close(wrench_fig)

    def close(self, exp_folder_path: str = "") -> None:
        if not self.enabled:
            return
        self._dump_pngs(exp_folder_path=exp_folder_path)


class ComplianceVLMPlotter:
    """Static plotting helpers for VLM affordance prediction outputs."""

    @staticmethod
    def _plot_frame(
        ax,
        origin: npt.NDArray[np.float32],
        rotation: npt.NDArray[np.float32],
        label: Optional[str] = None,
        length: float = 0.05,
    ) -> None:
        rotation_arr = np.asarray(rotation, dtype=np.float32)
        origin_arr = np.asarray(origin, dtype=np.float32)
        x_axis = rotation_arr[:, 0]
        y_axis = rotation_arr[:, 1]
        z_axis = rotation_arr[:, 2]

        ax.quiver(
            origin_arr[0],
            origin_arr[1],
            origin_arr[2],
            x_axis[0],
            x_axis[1],
            x_axis[2],
            color="r",
            length=length,
        )
        ax.quiver(
            origin_arr[0],
            origin_arr[1],
            origin_arr[2],
            y_axis[0],
            y_axis[1],
            y_axis[2],
            color="g",
            length=length,
        )
        ax.quiver(
            origin_arr[0],
            origin_arr[1],
            origin_arr[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="b",
            length=length,
        )
        if label:
            offset_dir = rotation_arr @ np.full(3, length * 0.3, dtype=np.float32)
            label_pos = origin_arr + offset_dir
            ax.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                label,
                fontsize=8,
                color="k",
            )

    @classmethod
    def visualize_results(
        cls,
        world_t_left_camera: npt.NDArray[np.float32],
        world_t_right_camera: npt.NDArray[np.float32],
        trajectories_by_site: Dict[str, Tuple[np.ndarray, ...]],
        robot_name: str = "toddlerbot_2xm",
        head_position: Optional[npt.NDArray[np.float32]] = None,
        head_orientation: Optional[npt.NDArray[np.float32]] = None,
        point_cloud_world: Optional[npt.NDArray[np.float32]] = None,
        point_colors: Optional[npt.NDArray[np.float32]] = None,
        contact_points_camera: Optional[Dict[str, npt.NDArray[np.float32]]] = None,
        contact_normals_camera: Optional[Dict[str, npt.NDArray[np.float32]]] = None,
    ):
        import matplotlib.pyplot as plt

        from vlm.affordance.plan_ee_pose import transform_normals, transform_points

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        if (
            robot_name == "leap_hand"
            and head_position is not None
            and head_orientation is not None
        ):
            try:
                head_rot_mat = R.from_quat(
                    head_orientation, scalar_first=True
                ).as_matrix()
            except Exception:
                head_rot_mat = np.eye(3, dtype=np.float32)
            cls._plot_frame(
                ax,
                origin=np.asarray(head_position, dtype=np.float32),
                rotation=head_rot_mat.astype(np.float32),
                label="Head",
                length=0.05,
            )

        cls._plot_frame(
            ax, origin=np.zeros(3, dtype=np.float32), rotation=np.eye(3), label="World"
        )
        cls._plot_frame(
            ax,
            origin=np.asarray(world_t_left_camera[:3, 3], dtype=np.float32),
            rotation=np.asarray(world_t_left_camera[:3, :3], dtype=np.float32),
            label="Left Camera",
        )
        cls._plot_frame(
            ax,
            origin=np.asarray(world_t_right_camera[:3, 3], dtype=np.float32),
            rotation=np.asarray(world_t_right_camera[:3, :3], dtype=np.float32),
            label="Right Camera",
        )

        palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        points_for_bounds: List[np.ndarray] = [
            np.zeros(3, dtype=np.float32),
            np.asarray(world_t_left_camera[:3, 3], dtype=np.float32),
            np.asarray(world_t_right_camera[:3, 3], dtype=np.float32),
        ]

        site_colors: Dict[str, str] = {}
        for idx, site_name in enumerate(trajectories_by_site.keys()):
            site_colors[site_name] = palette[idx % len(palette)]

        if contact_points_camera:
            for idx, (site_name, points_cam) in enumerate(
                contact_points_camera.items()
            ):
                points_world = transform_points(points_cam, world_t_left_camera)
                if points_world.size == 0:
                    continue
                color = site_colors.get(site_name, palette[idx % len(palette)])
                ax.scatter(
                    points_world[:, 0],
                    points_world[:, 1],
                    points_world[:, 2],
                    color=color,
                    s=18,
                    marker="o",
                    edgecolor="white",
                    linewidth=0.4,
                    depthshade=False,
                    label=f"{site_name} contact",
                )
                points_for_bounds.extend(points_world)

                if contact_normals_camera and site_name in contact_normals_camera:
                    normals_world = transform_normals(
                        contact_normals_camera[site_name], world_t_left_camera
                    )
                    if normals_world.shape[0] == points_world.shape[0]:
                        ax.quiver(
                            points_world[:, 0],
                            points_world[:, 1],
                            points_world[:, 2],
                            normals_world[:, 0],
                            normals_world[:, 1],
                            normals_world[:, 2],
                            color=color,
                            length=0.03,
                            normalize=True,
                            linewidth=0.6,
                        )

        for idx, (site_name, traj) in enumerate(trajectories_by_site.items()):
            if not isinstance(traj, tuple) or len(traj) < 4:
                continue
            positions = np.asarray(traj[2], dtype=np.float32)
            orientations = np.asarray(traj[3], dtype=np.float32)
            if positions.size == 0:
                continue
            color = site_colors.get(site_name, palette[idx % len(palette)])
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=color,
                linewidth=1.5,
                label=f"{site_name} path",
            )
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=color,
                s=10,
                alpha=0.6,
                depthshade=False,
            )

            start = positions[0]
            end = positions[-1]
            ax.scatter(
                start[0],
                start[1],
                start[2],
                color=color,
                s=60,
                edgecolor="white",
                linewidth=0.6,
                depthshade=False,
                label=f"{site_name} start",
            )
            ax.scatter(
                end[0],
                end[1],
                end[2],
                color=color,
                s=70,
                marker="X",
                edgecolor="white",
                linewidth=0.6,
                depthshade=False,
                label=f"{site_name} end",
            )

            if orientations.size >= 3:
                rotation_mats = (
                    R.from_rotvec(orientations).as_matrix().astype(np.float32)
                )
                sample_count = min(4, positions.shape[0])
                frame_indices = np.unique(
                    np.linspace(0, positions.shape[0] - 1, sample_count, dtype=int)
                )
                for frame_idx in frame_indices:
                    cls._plot_frame(
                        ax,
                        origin=positions[frame_idx],
                        rotation=rotation_mats[frame_idx],
                        length=0.025,
                    )

            label_pos = end + np.array([0.0, 0.0, 0.015], dtype=np.float32)
            ax.text(
                label_pos[0],
                label_pos[1],
                label_pos[2],
                site_name,
                fontsize=9,
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": color,
                    "alpha": 0.9,
                },
            )
            points_for_bounds.extend(positions)

        if point_cloud_world is not None and point_cloud_world.size > 0:
            points_for_bounds.append(point_cloud_world)
            colors = None
            if (
                point_colors is not None
                and point_colors.shape[0] == point_cloud_world.shape[0]
            ):
                colors = np.clip(point_colors, 0.0, 1.0)

            scatter_kwargs: Dict[str, object] = {
                "s": 2,
                "alpha": 0.4,
                "label": "Point Cloud",
            }
            if colors is not None:
                scatter_kwargs["c"] = colors
            else:
                scatter_kwargs["color"] = "dodgerblue"

            ax.scatter(
                point_cloud_world[:, 0],
                point_cloud_world[:, 1],
                point_cloud_world[:, 2],
                **scatter_kwargs,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        pts = np.vstack(points_for_bounds) if points_for_bounds else np.zeros((1, 3))
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        ranges = maxs - mins
        max_range = np.max(ranges)
        if max_range > 0:
            mids = (maxs + mins) / 2.0
            half = max_range / 2.0
            ax.set_xlim(mids[0] - half, mids[0] + half)
            ax.set_ylim(mids[1] - half, mids[1] + half)
            ax.set_zlim(mids[2] - half, mids[2] + half)
            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect((1.0, 1.0, 1.0))

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right")
        ax.set_title("Affordance Pose Planning")
        ax.grid(True)
        ax.view_init(elev=30, azim=135)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        return fig

    @staticmethod
    def plot_trajectory_profiles(
        trajectories_by_site: Optional[Dict[str, Tuple[np.ndarray, ...]]],
    ):
        import matplotlib.pyplot as plt

        if not trajectories_by_site:
            return None

        palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
        fig, axes = plt.subplots(6, 1, figsize=(9, 14))
        fig.subplots_adjust(hspace=0.32)

        for idx, (site_name, traj) in enumerate(trajectories_by_site.items()):
            if not isinstance(traj, tuple) or len(traj) < 3:
                continue
            color = palette[idx % len(palette)]
            t = np.asarray(traj[0], dtype=np.float64)
            stage = np.asarray(traj[1], dtype=np.float64) if len(traj) > 1 else None
            pos = np.asarray(traj[2], dtype=np.float64)
            orient = np.asarray(traj[3], dtype=np.float64) if len(traj) > 3 else None
            if (
                pos.ndim != 2
                or pos.shape[1] != 3
                or t.size == 0
                or pos.shape[0] != t.size
            ):
                continue
            if orient is not None and (
                orient.ndim != 2 or orient.shape[1] != 3 or orient.shape[0] != t.size
            ):
                orient = None

            vel = np.gradient(pos, t, axis=0) if t.size > 1 else np.zeros_like(pos)
            acc = np.gradient(vel, t, axis=0) if t.size > 1 else np.zeros_like(pos)

            axes[0].plot(
                pos[:, 0], pos[:, 1], color=color, linewidth=1.3, label=site_name
            )
            axes[0].scatter(
                pos[0, 0],
                pos[0, 1],
                color=color,
                s=40,
                edgecolor="white",
                linewidth=0.6,
                label=f"{site_name} start",
            )
            axes[0].scatter(
                pos[-1, 0],
                pos[-1, 1],
                color=color,
                s=50,
                marker="X",
                edgecolor="white",
                linewidth=0.6,
                label=f"{site_name} end",
            )

            axes[1].plot(
                t, pos[:, 0], color=color, linestyle="-", label=f"{site_name} px"
            )
            axes[1].plot(
                t, pos[:, 1], color=color, linestyle="--", label=f"{site_name} py"
            )
            axes[1].plot(
                t, pos[:, 2], color=color, linestyle=":", label=f"{site_name} pz"
            )

            if orient is not None:
                axes[2].plot(
                    t, orient[:, 0], color=color, linestyle="-", label=f"{site_name} rx"
                )
                axes[2].plot(
                    t,
                    orient[:, 1],
                    color=color,
                    linestyle="--",
                    label=f"{site_name} ry",
                )
                axes[2].plot(
                    t, orient[:, 2], color=color, linestyle=":", label=f"{site_name} rz"
                )

            axes[3].plot(
                t, vel[:, 0], color=color, linestyle="-", label=f"{site_name} vx"
            )
            axes[3].plot(
                t, vel[:, 1], color=color, linestyle="--", label=f"{site_name} vy"
            )
            axes[3].plot(
                t, vel[:, 2], color=color, linestyle=":", label=f"{site_name} vz"
            )

            axes[4].plot(
                t, acc[:, 0], color=color, linestyle="-", label=f"{site_name} ax"
            )
            axes[4].plot(
                t, acc[:, 1], color=color, linestyle="--", label=f"{site_name} ay"
            )
            axes[4].plot(
                t, acc[:, 2], color=color, linestyle=":", label=f"{site_name} az"
            )

            speed = np.linalg.norm(vel, axis=1)
            axes[5].plot(t, speed, color=color, label=f"{site_name} speed")
            accel_mag = np.linalg.norm(acc, axis=1)
            axes[5].plot(
                t,
                accel_mag,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"{site_name} accel",
            )
            if stage is not None and stage.shape == t.shape:
                axes[5].step(
                    t,
                    stage,
                    where="post",
                    color=color,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"{site_name} stage",
                )

        axes[0].set_title("Planned Trajectory (XY plane)")
        axes[0].set_xlabel("X [m]")
        axes[0].set_ylabel("Y [m]")
        axes[0].axis("equal")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        if trajectories_by_site:
            first_traj = next(iter(trajectories_by_site.values()))
            if isinstance(first_traj, tuple) and len(first_traj) >= 1:
                t_first = np.asarray(first_traj[0])
                if t_first.size > 0:
                    for ax in axes[1:]:
                        ax.set_xlim(t_first[0], t_first[-1])

        axes[1].set_title("Position vs Time")
        axes[1].set_ylabel("Position [m]")
        axes[1].grid(True, alpha=0.3)
        if any(line.get_visible() for line in axes[1].lines):
            axes[1].legend(loc="best")

        axes[2].set_title("Orientation (Rotation Vector) vs Time")
        axes[2].set_ylabel("Rotation [rad]")
        axes[2].grid(True, alpha=0.3)
        if any(line.get_visible() for line in axes[2].lines):
            axes[2].legend(loc="best")

        axes[3].set_title("Velocity vs Time")
        axes[3].set_ylabel("Velocity [m/s]")
        axes[3].grid(True, alpha=0.3)
        if any(line.get_visible() for line in axes[3].lines):
            axes[3].legend(loc="best")

        axes[4].set_title("Acceleration vs Time")
        axes[4].set_ylabel("Acceleration [m/s^2]")
        axes[4].grid(True, alpha=0.3)
        if any(line.get_visible() for line in axes[4].lines):
            axes[4].legend(loc="best")

        axes[5].set_title("Speed / Acceleration / Stage vs Time")
        axes[5].set_ylabel("Speed / Accel / Stage")
        axes[5].grid(True, alpha=0.3)
        if any(line.get_visible() for line in axes[5].lines):
            axes[5].legend(loc="best")
        axes[5].set_xlabel("Time [s]")

        fig.suptitle("Dense Trajectory Profiles", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    @staticmethod
    def load_pose_data(
        data_path: Path,
    ) -> Dict[str, Union[np.ndarray, Dict[str, Tuple[np.ndarray, ...]]]]:
        try:
            import joblib
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "plotting requires joblib. Please install it in your environment."
            ) from exc

        if not data_path.exists():
            raise FileNotFoundError(f"Pose data file not found: {data_path}")
        payload = joblib.load(data_path)
        required_keys = {
            "world_T_left_camera",
            "world_T_right_camera",
            "trajectory_by_site",
        }
        missing = required_keys - set(payload.keys())
        if missing:
            raise KeyError(f"Pose data missing keys: {', '.join(sorted(missing))}")

        result: Dict[str, Union[np.ndarray, Dict[str, Tuple[np.ndarray, ...]]]] = {
            "world_T_left_camera": np.asarray(
                payload["world_T_left_camera"], dtype=np.float32
            ),
            "world_T_right_camera": np.asarray(
                payload["world_T_right_camera"], dtype=np.float32
            ),
        }

        traj_by_site_raw = payload.get("trajectory_by_site", {})
        trajectories: Dict[str, Tuple[np.ndarray, ...]] = {}
        if isinstance(traj_by_site_raw, dict):
            for site_name, traj in traj_by_site_raw.items():
                if isinstance(traj, tuple):
                    trajectories[site_name] = tuple(np.asarray(comp) for comp in traj)
        result["trajectory_by_site"] = trajectories

        contact_points_camera = payload.get("contact_pos_camera")
        contact_normals_camera = payload.get("contact_normals_camera")
        if isinstance(contact_points_camera, dict):
            result["contact_pos_camera"] = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in contact_points_camera.items()
            }
        if isinstance(contact_normals_camera, dict):
            result["contact_normals_camera"] = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in contact_normals_camera.items()
            }

        return result

    @staticmethod
    def save_image_grid(
        prediction_dir: Path, desired_order: Sequence[str], plt_module
    ) -> Optional[Path]:
        image_paths = []
        for filename in desired_order:
            path = prediction_dir / filename
            if path.exists():
                image_paths.append(path)
            else:
                image_paths.append(None)

        if not any(path is not None for path in image_paths):
            return None

        rows, cols = 2, 4
        fig_images, axes = plt_module.subplots(
            rows, cols, figsize=(2.6 * cols, 2.4 * rows)
        )
        axes = axes.reshape(rows, cols)
        for idx, ax in enumerate(axes.flat):
            if idx < len(image_paths) and image_paths[idx] is not None:
                img = plt_module.imread(str(image_paths[idx]))
                ax.imshow(img)
                ax.set_title(desired_order[idx], fontsize=7, pad=2)
            else:
                ax.set_facecolor("#f0f0f0")
                ax.text(
                    0.5,
                    0.5,
                    "Missing",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="gray",
                )
            ax.axis("off")
        fig_images.subplots_adjust(
            left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.02, hspace=0.08
        )
        fig_images.suptitle("Affordance Debug Images", fontsize=10)
        out_path = prediction_dir / "debug_images_grid.png"
        fig_images.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.02)
        plt_module.close(fig_images)
        return out_path

    @classmethod
    def plot_prediction_results(cls, prediction_dir: Path) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("[AffordanceRun] Plot saving skipped: matplotlib is not installed.")
            return

        try:
            pose_data = cls.load_pose_data(prediction_dir / "trajectory.lz4")
        except FileNotFoundError:
            print(f"[AffordanceRun] No trajectory.lz4 to plot in {prediction_dir}.")
            return
        except Exception as exc:
            print(
                f"[AffordanceRun] Failed to load plot data from {prediction_dir}: {exc}"
            )
            return

        world_t_left_camera = np.asarray(
            pose_data["world_T_left_camera"], dtype=np.float32
        )
        world_t_right_camera = np.asarray(
            pose_data["world_T_right_camera"], dtype=np.float32
        )
        trajectories_by_site = pose_data.get("trajectory_by_site", {})
        trajectories_by_site = (
            trajectories_by_site if isinstance(trajectories_by_site, dict) else {}
        )
        contact_points_camera = pose_data.get("contact_pos_camera")
        contact_normals_camera = pose_data.get("contact_normals_camera")
        if not isinstance(contact_points_camera, dict):
            contact_points_camera = None
        if not isinstance(contact_normals_camera, dict):
            contact_normals_camera = None

        robot_name = "toddlerbot_2xm"
        head_position = None
        head_orientation = None
        args_path = prediction_dir / "args.json"
        if args_path.exists():
            try:
                args_payload = json.loads(args_path.read_text(encoding="utf-8"))
                if isinstance(args_payload, dict) and "robot_variant" in args_payload:
                    robot_name = str(args_payload["robot_variant"])
                if isinstance(args_payload, dict):
                    head_position = np.asarray(
                        args_payload.get("head_position", []), dtype=np.float32
                    )
                    head_orientation = np.asarray(
                        args_payload.get("head_orientation", []), dtype=np.float32
                    )
                    if head_position.size != 3:
                        head_position = None
                    if head_orientation.size != 4:
                        head_orientation = None
            except Exception as exc:
                print(f"[AffordanceRun] Warning: failed to read args.json: {exc}")

        point_cloud_world: Optional[np.ndarray] = None
        point_colors: Optional[np.ndarray] = None
        ply_files = sorted(glob.glob(str(prediction_dir / "*.ply")))
        if ply_files:
            try:
                import open3d as o3d
            except ModuleNotFoundError:
                o3d = None
            if o3d is not None:
                from vlm.affordance.plan_ee_pose import transform_points

                ply_path = ply_files[0]
                pcd = o3d.io.read_point_cloud(ply_path)
                points_cam = np.asarray(pcd.points, dtype=np.float32)
                if points_cam.size > 0:
                    point_cloud_world = transform_points(
                        points_cam, world_t_left_camera
                    )
                    colors = np.asarray(pcd.colors, dtype=np.float32)
                    if colors.shape == points_cam.shape and colors.size > 0:
                        point_colors = colors

        fig_pose = cls.visualize_results(
            world_t_left_camera=world_t_left_camera,
            world_t_right_camera=world_t_right_camera,
            robot_name=robot_name,
            head_position=head_position,
            head_orientation=head_orientation,
            trajectories_by_site=trajectories_by_site,
            point_cloud_world=point_cloud_world,
            point_colors=point_colors,
            contact_points_camera=contact_points_camera,
            contact_normals_camera=contact_normals_camera,
        )
        fig_pose.suptitle("Affordance Pose Visualization")
        pose_out = prediction_dir / "affordance_pose_plot.png"
        fig_pose.savefig(pose_out, dpi=170, bbox_inches="tight")
        plt.close(fig_pose)

        fig_traj = cls.plot_trajectory_profiles(trajectories_by_site)
        if fig_traj is not None:
            traj_out = prediction_dir / "trajectory_profiles.png"
            fig_traj.savefig(traj_out, dpi=170, bbox_inches="tight")
            plt.close(fig_traj)

        desired_order = [
            "left_raw.png",
            "right_raw.png",
            "segmentation_mask.png",
            "depth_map_vis.png",
            "left_rectified.png",
            "right_rectified.png",
            "candidate_points.png",
            "contact_points_overlay.png",
        ]
        grid_out = cls.save_image_grid(prediction_dir, desired_order, plt)

        print(f"[AffordanceRun] Saved plot: {pose_out}")
        if fig_traj is not None:
            print(
                f"[AffordanceRun] Saved plot: {prediction_dir / 'trajectory_profiles.png'}"
            )
        if grid_out is not None:
            print(f"[AffordanceRun] Saved plot: {grid_out}")
