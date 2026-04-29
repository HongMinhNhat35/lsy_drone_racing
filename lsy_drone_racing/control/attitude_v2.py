
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


# ============================================================
# SELECT CONTROLLER HERE
# ============================================================

# MODE = "CARROT"   # Try first
# MODE = "GATE"   # Try second
MODE = "TIMED"  # Try third


# ============================================================
# Shared constants
# ============================================================

GATE_Z_OFFSET = -0.10
OBSTACLE_CLEARANCE_RADIUS = 0.32

MAX_TILT_CARROT = 0.58
MAX_TILT_GATE = 0.55
MAX_TILT_TIMED = 0.62

YAW_DES = 0.0

NOMINAL_WAYPOINTS = np.array(
    [
        [-1.5, 0.75, 0.05],  # 0 start
        [-1.0, 0.55, 0.40],  # 1
        [0.0, 0.45, 0.70],   # 2 approach gate 0
        [0.5, 0.25, 0.70],   # 3 gate 0
        [1.3, -0.15, 0.90],  # 4 approach gate 1
        [1.05, 0.75, 1.20],  # 5 gate 1
        [0.65, 1.0, 1.20],   # 6
        [-0.2, -0.05, 0.60], # 7
        [-0.6, -0.2, 0.60],  # 8 approach gate 2
        [-1.0, -0.25, 0.70], # 9 gate 2
        [-1.5, -0.4, 0.70],  # 10
        [-1.5, -0.5, 1.20],  # 11
        [-1.0, -0.7, 1.20],  # 12 approach gate 3
        [-0.5, -0.65, 1.20], # 13
        [-0.2, -0.65, 1.20], # 14
        [0.0, -0.75, 1.20],  # 15 gate 3
        [0.5, -0.75, 1.20],  # 16 end
    ],
    dtype=np.float64,
)

GATE_WAYPOINT_IDX = {0: 3, 1: 5, 2: 9, 3: 15}


class AttitudeController(Controller):
    """Emergency attitude controller with three selectable strategies."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = config.env.freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = float(drone_params["mass"])
        self.thrust_min = float(drone_params["thrust_min"] * 4)
        self.thrust_max = float(drone_params["thrust_max"] * 4)

        self.g = 9.81

        self._tick = 0
        self._finished = False
        self._path_ready = False

        self._waypoints = NOMINAL_WAYPOINTS.copy()

        # For CARROT mode
        self._cum_lengths = None
        self._seg_lengths = None
        self._total_length = 0.0
        self._s_progress = 0.0
        self._gate_s = np.zeros(4)

        # For GATE mode
        self._target_idx = 1

        # For TIMED mode
        self._t_total = 8.8
        self._pos_spline = None
        self._vel_spline = None
        self._acc_spline = None

        self._build_path_from_obs(obs)

    # ============================================================
    # Shared helpers
    # ============================================================

    def _build_path_from_obs(self, obs: dict[str, NDArray[np.floating]]) -> None:
        waypoints = NOMINAL_WAYPOINTS.copy()

        if "pos" in obs:
            start_pos = np.asarray(obs["pos"], dtype=np.float64)
            if start_pos.shape == (3,) and np.all(np.isfinite(start_pos)):
                waypoints[0] = start_pos

        if "gates_pos" in obs:
            gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
            if gates_pos.ndim == 2 and gates_pos.shape[0] >= 4:
                for gate_i, wp_i in GATE_WAYPOINT_IDX.items():
                    gate = gates_pos[gate_i].copy()
                    if np.all(np.isfinite(gate)):
                        gate[2] += GATE_Z_OFFSET
                        waypoints[wp_i] = gate

        if "obstacles_pos" in obs:
            obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float64)
            if obstacles.size > 0:
                obstacles = obstacles.reshape(-1, 3)
                waypoints = self._shift_non_gate_waypoints(waypoints, obstacles)

        self._waypoints = waypoints

        self._rebuild_polyline()
        self._rebuild_timed_spline()

        self._target_idx = 1
        self._s_progress = 0.0
        self._path_ready = True

    def _shift_non_gate_waypoints(
        self,
        waypoints: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        shifted = waypoints.copy()
        gate_indices = set(GATE_WAYPOINT_IDX.values())

        for i in range(len(shifted)):
            if i in gate_indices:
                continue

            p = shifted[i].copy()

            for obs_pos in obstacles:
                if not np.all(np.isfinite(obs_pos)):
                    continue

                delta_xy = p[:2] - obs_pos[:2]
                dist_xy = float(np.linalg.norm(delta_xy))

                if dist_xy < OBSTACLE_CLEARANCE_RADIUS:
                    if dist_xy < 1e-6:
                        direction = np.array([1.0, 0.0])
                    else:
                        direction = delta_xy / dist_xy

                    p[:2] = obs_pos[:2] + direction * OBSTACLE_CLEARANCE_RADIUS

            shifted[i] = p

        return shifted

    def _rebuild_polyline(self) -> None:
        diffs = np.diff(self._waypoints, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        seg_lengths = np.maximum(seg_lengths, 1e-6)

        self._seg_lengths = seg_lengths
        self._cum_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        self._total_length = float(self._cum_lengths[-1])

        for gate_i, wp_i in GATE_WAYPOINT_IDX.items():
            self._gate_s[gate_i] = self._cum_lengths[wp_i]

    def _rebuild_timed_spline(self) -> None:
        diffs = np.diff(self._waypoints, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))

        total_len = max(float(cum_lengths[-1]), 1e-6)
        t_knots = cum_lengths / total_len * self._t_total

        for i in range(1, len(t_knots)):
            if t_knots[i] <= t_knots[i - 1]:
                t_knots[i] = t_knots[i - 1] + 1e-3

        self._pos_spline = CubicSpline(t_knots, self._waypoints, bc_type="clamped")
        self._vel_spline = self._pos_spline.derivative(1)
        self._acc_spline = self._pos_spline.derivative(2)

    def _sample_path(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        s = float(np.clip(s, 0.0, self._total_length))

        if s >= self._total_length:
            return self._waypoints[-1].copy(), np.zeros(3)

        seg_i = int(np.searchsorted(self._cum_lengths, s, side="right") - 1)
        seg_i = int(np.clip(seg_i, 0, len(self._seg_lengths) - 1))

        s0 = self._cum_lengths[seg_i]
        seg_len = self._seg_lengths[seg_i]
        u = float(np.clip((s - s0) / seg_len, 0.0, 1.0))

        p0 = self._waypoints[seg_i]
        p1 = self._waypoints[seg_i + 1]

        pos = p0 + u * (p1 - p0)
        tangent = (p1 - p0) / seg_len

        return pos, tangent

    def _closest_s_on_path(self, pos: np.ndarray) -> float:
        best_s = self._s_progress
        best_d = 1e9

        for i in range(len(self._seg_lengths)):
            p0 = self._waypoints[i]
            p1 = self._waypoints[i + 1]
            d = p1 - p0
            L2 = float(np.dot(d, d))

            if L2 < 1e-9:
                continue

            u = float(np.clip(np.dot(pos - p0, d) / L2, 0.0, 1.0))
            proj = p0 + u * d
            dist = float(np.linalg.norm(pos - proj))

            if dist < best_d:
                best_d = dist
                best_s = self._cum_lengths[i] + u * self._seg_lengths[i]

        return float(best_s)

    def _near_gate_scale(self, s: float) -> float:
        d = float(np.min(np.abs(self._gate_s - s)))
        if d < 0.28:
            return 0.72
        return 1.0

    def _force_to_action(
        self,
        force_world: np.ndarray,
        obs: dict[str, NDArray[np.floating]],
        max_tilt: float,
    ) -> np.ndarray:
        if not np.all(np.isfinite(force_world)):
            force_world = np.array([0.0, 0.0, self.drone_mass * self.g])

        force_norm = float(np.linalg.norm(force_world))
        if force_norm < 1e-6:
            force_world = np.array([0.0, 0.0, self.drone_mass * self.g])
            force_norm = float(np.linalg.norm(force_world))

        z_axis_desired = force_world / force_norm

        x_c_des = np.array([math.cos(YAW_DES), math.sin(YAW_DES), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_norm = float(np.linalg.norm(y_axis_desired))

        if y_norm < 1e-6:
            y_axis_desired = np.array([0.0, 1.0, 0.0])
        else:
            y_axis_desired /= y_norm

        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        euler[0] = np.clip(euler[0], -max_tilt, max_tilt)
        euler[1] = np.clip(euler[1], -max_tilt, max_tilt)
        euler[2] = 0.0

        # Use original-style thrust projection, since the original attitude controller
        # at least completed some runs. Then clip for safety.
        z_axis_current = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust = float(np.dot(force_world, z_axis_current))
        thrust = float(np.clip(thrust, self.thrust_min, self.thrust_max))

        return np.concatenate([euler, np.array([thrust])]).astype(np.float32)

    def _pd_force(
        self,
        des_pos: np.ndarray,
        des_vel: np.ndarray,
        obs: dict[str, NDArray[np.floating]],
        kp: np.ndarray,
        kd: np.ndarray,
        acc_ff: np.ndarray | None = None,
    ) -> np.ndarray:
        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)

        pos_error = des_pos - pos
        vel_error = des_vel - vel

        # Prevent one huge reference jump from asking for insane tilt.
        xy_norm = float(np.linalg.norm(pos_error[:2]))
        if xy_norm > 1.0:
            pos_error[:2] *= 1.0 / xy_norm

        pos_error[2] = float(np.clip(pos_error[2], -0.65, 0.65))

        force = kp * pos_error + kd * vel_error

        if acc_ff is not None:
            force += 0.20 * self.drone_mass * acc_ff

        force[2] += self.drone_mass * self.g

        return force

    # ============================================================
    # Controller 1: CARROT
    # ============================================================

    def _compute_carrot(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Pure-pursuit attitude controller.

        Tracks a point slightly ahead on the path based on the drone's actual
        position, not a fixed clock. This should be the most robust fast option.
        """
        pos = np.asarray(obs["pos"], dtype=np.float64)

        closest_s = self._closest_s_on_path(pos)
        self._s_progress = max(self._s_progress, closest_s)

        gate_scale = self._near_gate_scale(self._s_progress)

        base_speed = 1.95
        if self._s_progress > self._gate_s[2]:
            base_speed = 1.75

        speed = base_speed * gate_scale

        lookahead = 0.38
        if gate_scale < 1.0:
            lookahead = 0.25

        des_s = min(self._s_progress + lookahead, self._total_length)
        des_pos, tangent = self._sample_path(des_s)
        des_vel = tangent * speed

        kp = np.array([1.15, 1.15, 2.05])
        kd = np.array([0.70, 0.70, 0.85])

        force = self._pd_force(des_pos, des_vel, obs, kp, kd)
        action = self._force_to_action(force, obs, MAX_TILT_CARROT)

        if self._s_progress >= self._total_length - 0.05:
            self._finished = True

        return action

    # ============================================================
    # Controller 2: GATE
    # ============================================================

    def _compute_gate(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Sequential waypoint/gate pursuer.

        Less elegant than CARROT, but sometimes better if the path projection
        logic behaves badly. It explicitly chases one waypoint at a time.
        """
        pos = np.asarray(obs["pos"], dtype=np.float64)

        self._target_idx = int(np.clip(self._target_idx, 1, len(self._waypoints) - 1))

        target = self._waypoints[self._target_idx]
        dist = float(np.linalg.norm(target - pos))

        gate_indices = set(GATE_WAYPOINT_IDX.values())

        if self._target_idx in gate_indices:
            switch_dist = 0.20
        else:
            switch_dist = 0.36

        if dist < switch_dist and self._target_idx < len(self._waypoints) - 1:
            self._target_idx += 1
            target = self._waypoints[self._target_idx]
            dist = float(np.linalg.norm(target - pos))

        direction = target - pos
        direction_norm = float(np.linalg.norm(direction))

        if direction_norm < 1e-6:
            des_vel = np.zeros(3)
        else:
            direction = direction / direction_norm

            if self._target_idx in gate_indices:
                speed = 1.40
            else:
                speed = 1.95

            if self._target_idx >= 10:
                speed = min(speed, 1.65)

            des_vel = direction * speed

        des_pos = target

        kp = np.array([0.95, 0.95, 1.85])
        kd = np.array([0.62, 0.62, 0.82])

        force = self._pd_force(des_pos, des_vel, obs, kp, kd)
        action = self._force_to_action(force, obs, MAX_TILT_GATE)

        if self._target_idx >= len(self._waypoints) - 1 and dist < 0.25:
            self._finished = True

        return action

    # ============================================================
    # Controller 3: TIMED
    # ============================================================

    def _compute_timed(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Fast timed spline attitude controller.

        This is the high-risk sub-9 attempt. It follows an 8.8 s spline.
        """
        t = min(self._tick / self._freq, self._t_total)

        if t >= self._t_total:
            self._finished = True

        des_pos = self._pos_spline(t)
        des_vel = self._vel_spline(t)
        des_acc = self._acc_spline(t)

        kp = np.array([0.85, 0.85, 1.75])
        kd = np.array([0.50, 0.50, 0.72])

        force = self._pd_force(des_pos, des_vel, obs, kp, kd, acc_ff=des_acc)
        action = self._force_to_action(force, obs, MAX_TILT_TIMED)

        return action

    # ============================================================
    # Main API
    # ============================================================

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        if not self._path_ready:
            self._build_path_from_obs(obs)

        if MODE == "CARROT":
            return self._compute_carrot(obs)

        if MODE == "GATE":
            return self._compute_gate(obs)

        if MODE == "TIMED":
            return self._compute_timed(obs)

        raise ValueError(f"Unknown MODE={MODE!r}. Use CARROT, GATE, or TIMED.")

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
        self._tick = 0
        self._finished = False
        self._path_ready = False
        self._target_idx = 1
        self._s_progress = 0.0