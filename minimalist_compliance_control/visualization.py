"""Matplotlib-based compliance/wrench logging and end-of-run PNG export."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import numpy.typing as npt


class CompliancePlotter:
    """Collects data online and dumps PNGs at shutdown.

    This class intentionally avoids per-step plotting so the control loop does not
    spend time in Matplotlib. PNGs are generated once in ``close()``.
    """

    def __init__(
        self,
        site_names: Sequence[str],
        enabled: bool = True,
        output_dir: Optional[str] = None,
    ) -> None:
        self.site_names = [str(name) for name in site_names]
        self.enabled = bool(enabled)
        self.output_dir = output_dir
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
        wrenches: Dict[str, npt.NDArray[np.float32]],
        applied_site_forces: Optional[npt.NDArray[np.float32]],
        wrench_sim: Any,
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
        obs = np.zeros((num_sites, 6), dtype=np.float64)
        for idx, site in enumerate(self.site_names):
            site_id = wrench_sim.site_ids[site]
            obs[idx, :3] = np.asarray(
                wrench_sim.data.site_xpos[site_id], dtype=np.float64
            )
            rotmat = np.asarray(
                wrench_sim.data.site_xmat[site_id], dtype=np.float64
            ).reshape(3, 3)
            obs[idx, 3:6] = self._mat_to_rotvec(rotmat)
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

    def _dump_pngs(self) -> None:
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
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
        pose_fig.savefig(os.path.join(self.output_dir, "compliance_ref.png"), dpi=150)
        wrench_fig.savefig(
            os.path.join(self.output_dir, "estimated_wrench.png"), dpi=150
        )
        plt.close(pose_fig)
        plt.close(wrench_fig)

    def close(self) -> None:
        if not self.enabled:
            return
        self._dump_pngs()
