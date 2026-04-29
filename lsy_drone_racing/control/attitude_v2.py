"""Fast Level 2 attitude controller.

Emergency Try 3:
- Uses 4D attitude interface: [roll, pitch, yaw, collective_thrust].
- Uses Level 2 gate-style waypoints instead of the old 10-point attitude path.
- Updates gate targets once from observed gate poses.
- Uses a fast 7.8 s trajectory.
- Uses PD + acceleration feedforward + gravity compensation.
- Clips tilt and thrust to avoid insane commands.

This is intended as a last high-speed attempt. The original attitude controller
you tested was still timed at 15 s, so it was not a real sub-8 attempt.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------
# Timing and control parameters
# ---------------------------------------------------------------------

TRAJECTORY_TIME_S = 7.8

# Slightly conservative tilt. If it is too slow but stable, try 0.75.
MAX_TILT_RAD = 0.70

# Gate center offset. Your tests showed that removing this globally was worse.
GATE_Z_OFFSET = -0.10

# If the drone falls behind badly, slow the virtual trajectory a little.
# This prevents instant crashes, but still mostly keeps sub-8 behavior.
ERROR_SLOWDOWN_START_M = 0.55
ERROR_SLOWDOWN_FULL_M = 1.20
MIN_TIME_SCALE = 0.70


# ---------------------------------------------------------------------
# Level 2 path
# ---------------------------------------------------------------------

NOMINAL_WAYPOINTS = np.array(
    [
        [-1.5, 0.75, 0.05],  # 0 start
        [-1.0, 0.55, 0.40],  # 1
        [0.0, 0.45, 0.70],  # 2 approach gate 0
        [0.5, 0.25, 0.70],  # 3 gate 0 center
        [1.3, -0.15, 0.90],  # 4 approach gate 1
        [1.05, 0.75, 1.20],  # 5 gate 1 center
        [0.65, 1.0, 1.20],  # 6
        [-0.2, -0.05, 0.60],  # 7
        [-0.6, -0.2, 0.60],  # 8 approach gate 2
        [-1.0, -0.25, 0.70],  # 9 gate 2 center
        [-1.5, -0.4, 0.70],  # 10
        [-1.5, -0.5, 1.20],  # 11
        [-1.0, -0.7, 1.20],  # 12 approach gate 3
        [-0.5, -0.65, 1.20],  # 13
        [-0.2, -0.65, 1.20],  # 14
        [0.0, -0.75, 1.20],  # 15 gate 3 center
        [0.5, -0.75, 1.20],  # 16 end
    ],
    dtype=np.float64,
)

GATE_WAYPOINT_IDX = {0: 3, 1: 5, 2: 9, 3: 15}


class AttitudeController(Controller):
    """Fast autonomous attitude controller for Level 2."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = float(drone_params["mass"])
        self.thrust_min = float(drone_params["thrust_min"] * 4)
        self.thrust_max = float(drone_params["thrust_max"] * 4)

        self.g = 9.81

        # Force-domain PD gains. These are intentionally stronger than the original
        # 15-second controller, but not maxed out.
        self.kp = np.array([0.95, 0.95, 1.85], dtype=np.float64)
        self.kd = np.array([0.48, 0.48, 0.72], dtype=np.float64)

        # No integral for racing. Integral windup is bad for aggressive trajectories.
        self.ki = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.i_error = np.zeros(3, dtype=np.float64)

        self._tick = 0
        self._finished = False
        self._last_yaw = 0.0
        self._t_ref = 0.0
        self._last_time_scale = 1.0

        self._waypoints = NOMINAL_WAYPOINTS.copy()
        self._initialize_path_from_obs(obs)
        self._build_spline()

    # ------------------------------------------------------------------
    # Path setup
    # ------------------------------------------------------------------

    def _initialize_path_from_obs(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Use observed gate positions once at reset."""
        waypoints = NOMINAL_WAYPOINTS.copy()

        if "pos" in obs:
            pos = np.asarray(obs["pos"], dtype=np.float64)
            if pos.shape == (3,) and np.all(np.isfinite(pos)):
                waypoints[0] = pos

        if "gates_pos" in obs:
            gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
            if gates_pos.ndim == 2 and gates_pos.shape[0] >= 4:
                for gate_i, wp_i in GATE_WAYPOINT_IDX.items():
                    gate = gates_pos[gate_i].copy()
                    if np.all(np.isfinite(gate)):
                        gate[2] += GATE_Z_OFFSET
                        waypoints[wp_i] = gate

        self._waypoints = waypoints

    def _build_spline(self) -> None:
        """Build a distance-retimed spline, then scale total time to target."""
        diffs = np.diff(self._waypoints, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))

        total_length = max(float(cum_lengths[-1]), 1e-6)
        t_knots = cum_lengths / total_length * TRAJECTORY_TIME_S

        # Make sure there are no duplicate timestamps.
        for i in range(1, len(t_knots)):
            if t_knots[i] <= t_knots[i - 1]:
                t_knots[i] = t_knots[i - 1] + 1e-3

        self._t_total = float(t_knots[-1])
        self._t_knots = t_knots

        self._pos_spline = CubicSpline(t_knots, self._waypoints, bc_type="clamped")
        self._vel_spline = self._pos_spline.derivative(1)
        self._acc_spline = self._pos_spline.derivative(2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _closest_angle(angle: float, reference: float) -> float:
        return float(reference + ((angle - reference + np.pi) % (2.0 * np.pi) - np.pi))

    def _compute_time_scale(self, obs: dict[str, NDArray[np.floating]], des_pos: np.ndarray) -> float:
        err = float(np.linalg.norm(des_pos - obs["pos"]))

        if err <= ERROR_SLOWDOWN_START_M:
            scale = 1.0
        elif err >= ERROR_SLOWDOWN_FULL_M:
            scale = MIN_TIME_SCALE
        else:
            alpha = (err - ERROR_SLOWDOWN_START_M) / (
                ERROR_SLOWDOWN_FULL_M - ERROR_SLOWDOWN_START_M
            )
            scale = (1.0 - alpha) + alpha * MIN_TIME_SCALE

        scale = 0.8 * self._last_time_scale + 0.2 * scale
        self._last_time_scale = float(scale)
        return float(scale)

    def _compute_yaw_and_rate(
        self,
        des_vel: np.ndarray,
        des_acc: np.ndarray,
    ) -> tuple[float, float]:
        vx, vy = float(des_vel[0]), float(des_vel[1])
        ax, ay = float(des_acc[0]), float(des_acc[1])

        speed_xy_sq = vx * vx + vy * vy
        if speed_xy_sq < 0.05 * 0.05:
            return self._wrap_to_pi(self._last_yaw), 0.0

        raw_yaw = float(np.arctan2(vy, vx))
        yaw = self._closest_angle(raw_yaw, self._last_yaw)
        yaw_rate = (vx * ay - vy * ax) / speed_xy_sq

        self._last_yaw = yaw
        return self._wrap_to_pi(yaw), float(np.clip(yaw_rate, -2.5, 2.5))

    def _force_to_euler_and_thrust(
        self,
        target_force_world: np.ndarray,
        des_yaw: float,
    ) -> tuple[np.ndarray, float]:
        norm_force = float(np.linalg.norm(target_force_world))
        if norm_force < 1e-6 or not np.isfinite(norm_force):
            target_force_world = np.array([0.0, 0.0, self.drone_mass * self.g])
            norm_force = float(np.linalg.norm(target_force_world))

        z_axis_desired = target_force_world / norm_force

        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0], dtype=np.float64)

        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_norm = float(np.linalg.norm(y_axis_desired))

        if y_norm < 1e-6:
            y_axis_desired = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            y_axis_desired /= y_norm

        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack(
            [x_axis_desired, y_axis_desired, z_axis_desired]
        ).T

        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        euler_desired[0] = np.clip(euler_desired[0], -MAX_TILT_RAD, MAX_TILT_RAD)
        euler_desired[1] = np.clip(euler_desired[1], -MAX_TILT_RAD, MAX_TILT_RAD)

        # Use force magnitude, not projection onto current z-axis. Projection can
        # under-command thrust while the drone is tilted and already lagging.
        thrust_desired = float(np.clip(norm_force, self.thrust_min, self.thrust_max))

        return euler_desired, thrust_desired

    # ------------------------------------------------------------------
    # Controller API
    # ------------------------------------------------------------------

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        t_probe = min(self._t_ref, self._t_total)
        des_pos_probe = self._pos_spline(t_probe)

        time_scale = self._compute_time_scale(obs, des_pos_probe)

        self._t_ref = min(self._t_ref + time_scale / self._freq, self._t_total)
        t = min(self._t_ref, self._t_total)

        if t >= self._t_total:
            self._finished = True

        des_pos = self._pos_spline(t)
        des_vel = self._vel_spline(t) * time_scale
        des_acc = self._acc_spline(t) * (time_scale * time_scale)

        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        des_yaw, _ = self._compute_yaw_and_rate(des_vel, des_acc)

        # Desired force in world frame.
        target_force = np.zeros(3, dtype=np.float64)
        target_force += self.kp * pos_error
        target_force += self.kd * vel_error
        target_force += self.drone_mass * des_acc
        target_force[2] += self.drone_mass * self.g

        euler_desired, thrust_desired = self._force_to_euler_and_thrust(
            target_force,
            des_yaw,
        )

        action = np.concatenate(
            [euler_desired, np.array([thrust_desired], dtype=np.float64)]
        ).astype(np.float32)

        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self.i_error[:] = 0.0
        self._tick = 0
        self._finished = False
        self._last_yaw = 0.0
        self._t_ref = 0.0
        self._last_time_scale = 1.0