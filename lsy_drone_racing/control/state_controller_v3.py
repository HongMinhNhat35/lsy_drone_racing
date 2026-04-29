"""State controller with original spline path plus virtual-time lag governor.

Emergency Try 2:
- Restores the original state_controller_v2 structure.
- Keeps the 13D state action interface.
- Keeps the original waypoint path and gate z-offset.
- Adds a virtual time variable so the trajectory slows down if the drone falls behind.
- Does not use the failed polyline controller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.controller import Controller

try:
    from crazyflow.sim.visualize import draw_line, draw_points

    HAS_VIZ = True
except ImportError:
    draw_line = None
    draw_points = None
    HAS_VIZ = False

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# ---------------------------------------------------------------------
# Tunable trajectory parameters
# ---------------------------------------------------------------------

TARGET_SPEED_MPS = 1.35
MAX_SPEED_MPS = 1.90
MAX_ACCEL_MPS2 = 5.0
MIN_SEGMENT_TIME = 0.20
RETIMING_SAMPLES = 160
RETIMING_ITERS = 4


# ---------------------------------------------------------------------
# Virtual-time governor
# ---------------------------------------------------------------------

# If tracking error is small, trajectory time advances normally.
TRACKING_ERROR_SLOWDOWN_START_M = 0.30

# If tracking error reaches this, trajectory time advances slowly.
TRACKING_ERROR_SLOWDOWN_FULL_M = 0.90

# Minimum fraction of normal trajectory-time speed.
MIN_TIME_SCALE = 0.35

# After passing the third gate region, be slightly more conservative.
TAIL_TIME_SCALE = 0.85


# ---------------------------------------------------------------------
# Gate and obstacle parameters
# ---------------------------------------------------------------------

NOMINAL_GATE_POS = np.array(
    [
        [0.5, 0.25, 0.7],
        [1.05, 0.75, 1.2],
        [-1.0, -0.25, 0.7],
        [0.0, -0.75, 1.2],
    ],
    dtype=np.float64,
)

GATE_WAYPOINT_IDX = {0: 3, 1: 5, 2: 9, 3: 15}

# Keep this. Your tests showed changing this globally made things worse.
GATE_Z_OFFSET = -0.10

GATE_UPDATE_EPS = 0.01

OBSTACLE_CLEARANCE_RADIUS = 0.32

LOCK_PAST_WAYPOINT_MARGIN_S = 0.15


# ---------------------------------------------------------------------
# Yaw parameters
# ---------------------------------------------------------------------

YAW_MIN_SPEED = 0.05


# ---------------------------------------------------------------------
# Nominal path
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


class StateController(Controller):
    """Adaptive spline state controller with virtual-time slowdown."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = config.env.freq

        self._nominal_waypoints = NOMINAL_WAYPOINTS.copy()
        self._gate_corrected_waypoints = self._nominal_waypoints.copy()
        self._waypoints = self._gate_corrected_waypoints.copy()

        self._gate_updated = [False] * len(NOMINAL_GATE_POS)

        self._tick = 0
        self._finished = False
        self._last_yaw = 0.0

        # Virtual trajectory time. This replaces tick/freq for reference progress.
        self._t_ref = 0.0
        self._last_time_scale = 1.0

        self._rebuild_spline()

    # ------------------------------------------------------------------
    # Small utility functions
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _closest_angle(angle: float, reference: float) -> float:
        return float(reference + ((angle - reference + np.pi) % (2.0 * np.pi) - np.pi))

    def _elapsed_time(self) -> float:
        return min(self._t_ref, self._t_total)

    def _is_waypoint_past(self, wp_idx: int, t_now: float) -> bool:
        if not hasattr(self, "_t_knots"):
            return False
        return self._t_knots[wp_idx] < t_now - LOCK_PAST_WAYPOINT_MARGIN_S

    # ------------------------------------------------------------------
    # Spline timing
    # ------------------------------------------------------------------

    def _make_time_knots(self, waypoints: NDArray[np.floating]) -> NDArray[np.floating]:
        segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

        segment_times = np.maximum(
            segment_lengths / TARGET_SPEED_MPS,
            MIN_SEGMENT_TIME,
        )

        return np.concatenate(([0.0], np.cumsum(segment_times)))

    def _rebuild_spline(self):
        t_knots = self._make_time_knots(self._waypoints)

        for _ in range(RETIMING_ITERS):
            spline = CubicSpline(t_knots, self._waypoints, bc_type="clamped")

            t_samples = np.linspace(t_knots[0], t_knots[-1], RETIMING_SAMPLES)

            vel = spline.derivative(1)(t_samples)
            acc = spline.derivative(2)(t_samples)

            max_speed = float(np.max(np.linalg.norm(vel, axis=1)))
            max_accel = float(np.max(np.linalg.norm(acc, axis=1)))

            speed_scale = max_speed / MAX_SPEED_MPS if max_speed > MAX_SPEED_MPS else 1.0
            accel_scale = (
                np.sqrt(max_accel / MAX_ACCEL_MPS2)
                if max_accel > MAX_ACCEL_MPS2
                else 1.0
            )

            scale = max(speed_scale, accel_scale)

            if scale <= 1.001:
                break

            t_knots = t_knots * (scale * 1.02)

        self._t_knots = t_knots
        self._t_total = float(t_knots[-1])

        self._spline = CubicSpline(self._t_knots, self._waypoints, bc_type="clamped")
        self._vel_spline = self._spline.derivative(1)
        self._acc_spline = self._spline.derivative(2)

    # ------------------------------------------------------------------
    # Gate correction
    # ------------------------------------------------------------------

    def _update_gate_waypoints(self, obs: dict) -> bool:
        if "gates_pos" not in obs:
            return False

        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)

        if gates_pos.ndim != 2 or gates_pos.shape[0] < len(NOMINAL_GATE_POS):
            return False

        t_now = self._elapsed_time()
        changed = False

        for gate_i, wp_i in GATE_WAYPOINT_IDX.items():
            if self._is_waypoint_past(wp_i, t_now):
                continue

            observed_pos = gates_pos[gate_i].copy()

            if not np.all(np.isfinite(observed_pos)):
                continue

            observed_pos[2] += GATE_Z_OFFSET

            current_target = self._gate_corrected_waypoints[wp_i]
            shift = np.linalg.norm(observed_pos - current_target)

            if shift > GATE_UPDATE_EPS:
                self._gate_corrected_waypoints[wp_i] = observed_pos
                self._gate_updated[gate_i] = True
                changed = True

        return changed

    # ------------------------------------------------------------------
    # Obstacle avoidance
    # ------------------------------------------------------------------

    def _shift_waypoint_from_obstacles(
        self,
        waypoint: NDArray[np.floating],
        obs_positions: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        shifted = waypoint.copy()

        for obs_pos in obs_positions:
            if not np.all(np.isfinite(obs_pos)):
                continue

            delta_xy = shifted[:2] - obs_pos[:2]
            dist_xy = np.linalg.norm(delta_xy)

            if dist_xy < OBSTACLE_CLEARANCE_RADIUS:
                if dist_xy < 1e-6:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = delta_xy / dist_xy

                shifted[:2] = obs_pos[:2] + direction * OBSTACLE_CLEARANCE_RADIUS

        return shifted

    def _update_obstacle_avoidance(self, obs: dict) -> bool:
        if "obstacles_pos" not in obs:
            return False

        obs_positions = np.asarray(obs["obstacles_pos"], dtype=np.float64)

        if obs_positions.size == 0:
            return False

        obs_positions = obs_positions.reshape(-1, 3)

        t_now = self._elapsed_time()

        candidate_waypoints = self._gate_corrected_waypoints.copy()

        for wp_i in range(len(candidate_waypoints)):
            if wp_i in GATE_WAYPOINT_IDX.values():
                continue

            if self._is_waypoint_past(wp_i, t_now):
                candidate_waypoints[wp_i] = self._waypoints[wp_i]
                continue

            candidate_waypoints[wp_i] = self._shift_waypoint_from_obstacles(
                candidate_waypoints[wp_i],
                obs_positions,
            )

        if not np.allclose(candidate_waypoints, self._waypoints, atol=1e-4):
            self._waypoints = candidate_waypoints
            return True

        return False

    def _update_path_from_observation(self, obs: dict):
        gates_changed = self._update_gate_waypoints(obs)
        obstacles_changed = self._update_obstacle_avoidance(obs)

        if gates_changed or obstacles_changed:
            # Preserve current virtual progress fraction after rebuilding.
            old_t_total = getattr(self, "_t_total", None)
            old_t_ref = self._t_ref

            self._rebuild_spline()

            if old_t_total is not None and old_t_total > 1e-6:
                frac = np.clip(old_t_ref / old_t_total, 0.0, 1.0)
                self._t_ref = float(frac * self._t_total)

    # ------------------------------------------------------------------
    # Virtual-time control
    # ------------------------------------------------------------------

    def _compute_time_scale(self, obs: dict, des_pos: NDArray[np.floating]) -> float:
        tracking_error = float(np.linalg.norm(des_pos - obs["pos"]))

        if tracking_error <= TRACKING_ERROR_SLOWDOWN_START_M:
            scale = 1.0
        elif tracking_error >= TRACKING_ERROR_SLOWDOWN_FULL_M:
            scale = MIN_TIME_SCALE
        else:
            alpha = (tracking_error - TRACKING_ERROR_SLOWDOWN_START_M) / (
                TRACKING_ERROR_SLOWDOWN_FULL_M - TRACKING_ERROR_SLOWDOWN_START_M
            )
            scale = (1.0 - alpha) + alpha * MIN_TIME_SCALE

        # Be more conservative after gate 2 / before final gate.
        if hasattr(self, "_t_knots") and self._t_ref >= self._t_knots[9]:
            scale = min(scale, TAIL_TIME_SCALE)

        # Smooth the time-scale itself to avoid jumps in desired velocity.
        scale = 0.85 * self._last_time_scale + 0.15 * scale
        self._last_time_scale = float(scale)

        return float(scale)

    # ------------------------------------------------------------------
    # Yaw calculation
    # ------------------------------------------------------------------

    def _compute_yaw_and_rate(
        self,
        des_vel: NDArray[np.floating],
        des_acc: NDArray[np.floating],
    ) -> tuple[float, float]:
        vx, vy = float(des_vel[0]), float(des_vel[1])
        ax, ay = float(des_acc[0]), float(des_acc[1])

        speed_xy_sq = vx * vx + vy * vy

        if speed_xy_sq < YAW_MIN_SPEED * YAW_MIN_SPEED:
            return self._wrap_to_pi(self._last_yaw), 0.0

        raw_yaw = float(np.arctan2(vy, vx))
        continuous_yaw = self._closest_angle(raw_yaw, self._last_yaw)

        yaw_rate = (vx * ay - vy * ax) / speed_xy_sq

        self._last_yaw = continuous_yaw

        return self._wrap_to_pi(continuous_yaw), float(yaw_rate)

    # ------------------------------------------------------------------
    # Main controller API
    # ------------------------------------------------------------------

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        self._update_path_from_observation(obs)

        # First sample at current virtual time.
        t = min(self._t_ref, self._t_total)

        if t >= self._t_total:
            self._finished = True

        des_pos_base = self._spline(t)

        # Slow reference time if drone is lagging.
        time_scale = self._compute_time_scale(obs, des_pos_base)

        # Advance virtual trajectory time here, not in step_callback.
        self._t_ref = min(self._t_ref + time_scale / self._freq, self._t_total)
        t = min(self._t_ref, self._t_total)

        des_pos = self._spline(t)

        # Time-warped derivatives. This keeps desired velocity/acceleration consistent
        # with the slowed virtual progress.
        des_vel = self._vel_spline(t) * time_scale
        des_acc = self._acc_spline(t) * (time_scale * time_scale)

        des_yaw, des_yaw_rate = self._compute_yaw_and_rate(des_vel, des_acc)

        action = np.concatenate(
            (
                des_pos,
                des_vel,
                des_acc,
                np.array([des_yaw]),
                np.array([0.0, 0.0, des_yaw_rate]),
            ),
            dtype=np.float32,
        )

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
        self._tick = 0
        self._finished = False
        self._last_yaw = 0.0

        self._t_ref = 0.0
        self._last_time_scale = 1.0

        self._gate_updated = [False] * len(NOMINAL_GATE_POS)

        self._gate_corrected_waypoints = self._nominal_waypoints.copy()
        self._waypoints = self._gate_corrected_waypoints.copy()

        self._rebuild_spline()

    def render_callback(self, sim: "Sim"):
        if not HAS_VIZ:
            return

        t_now = self._elapsed_time()

        if t_now < self._t_total:
            t_vals = np.linspace(t_now, self._t_total, 120)
        else:
            t_vals = np.array([self._t_total, self._t_total])

        trajectory = self._spline(t_vals)
        setpoint = self._spline(t_now).reshape(1, -1)

        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)