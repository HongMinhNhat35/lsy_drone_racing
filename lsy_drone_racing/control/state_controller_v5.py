"""State controller that follows an adaptive, faster spline trajectory.

This version improves the baseline controller by:
1. Retiming the trajectory based on waypoint distances.
2. Providing desired position, velocity, acceleration, yaw, and yaw-rate.
3. Updating gate waypoints from observed gate positions.
4. Shifting non-gate waypoints away from obstacles.
5. Avoiding cumulative waypoint drift by always recomputing from a clean base path.
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

# Higher = faster trajectory. If the drone starts missing gates, reduce this.
TARGET_SPEED_MPS = 4.5

# Hard upper bound used when retiming the spline.
# The controller stretches the trajectory if the spline exceeds this.
MAX_SPEED_MPS = 6.0

# Hard acceleration limit used when retiming the spline.
# If the spline asks for too much acceleration, the trajectory is slowed down.
MAX_ACCEL_MPS2 = 7.0

# Prevents very short waypoint segments from becoming too aggressive.
MIN_SEGMENT_TIME = 0.20

# Number of samples used to estimate max speed and acceleration during retiming.
RETIMING_SAMPLES = 400

# Number of retiming iterations.
RETIMING_ITERS = 15


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

# Waypoint index corresponding to each gate center.
GATE_WAYPOINT_IDX = {0: 3, 1: 5, 2: 9, 3: 11}

# Fly slightly lower than the detected gate center.
# Set this to 0.0 if you want to aim exactly at the center.
GATE_Z_OFFSET = -0.10

# Ignore very tiny gate shifts
GATE_UPDATE_EPS = 0.01

# Obstacle clearance radius in x-y plane.
# Actual obstacles may be smaller, but this creates a safety buffer.
OBSTACLE_CLEARANCE_RADIUS = 0.35

# Do not keep modifying waypoints that are already in the past.
LOCK_PAST_WAYPOINT_MARGIN_S = 0.15


# ---------------------------------------------------------------------
# Yaw parameters
# ---------------------------------------------------------------------

# Below this x-y speed, yaw becomes numerically unstable.
YAW_MIN_SPEED = 0.05


# --------------------------------------
# Nominal path
# --------------------------------------------------------

NOMINAL_WAYPOINTS = np.array(
    [
        [-1.5, 0.75, 0.05],  # 0 start
        [-1.0, 0.55, 0.40],  # 1
        [0.0, 0.25, 0.70],  # 2 approach gate 0
        [0.55, 0.20, 0.70],  # 3 gate 0 center
        # [1.15, 0.0, 0.70],
        [1.4, 0.1, 0.90],  # 4 approach gate 1
        [1.1, 0.78, 1.20],  # 5 gate 1 center
        [0.40, 0.75, 1.20],  # 6
        [-0.15, 0.0, 0.60],  # `7
        [-0.6, -0.2, 0.60],  # 8 approach gate 2
        [-1.0, -0.25, 0.70],   # 9 gate 2 center

        [-0.5, -0.45, 1.0],  # 13

        [0.0, -0.75, 1.20],  # 15 gate 3 center
        [0.5, -0.75, 1.20],  # 16 end
    ],
    dtype=np.float64,
)


class StateController(Controller):
    """Adaptive state controller following a retimed cubic spline trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller.

        Args:
            obs: Initial environment observation.
            info: Initial environment info.
            config: Race/environment configuration.
        """
        super().__init__(obs, info, config)

        self._freq = config.env.freq

        # This is the clean nominal path.
        # It never accumulates obstacle shifts.
        self._nominal_waypoints = NOMINAL_WAYPOINTS.copy()

        # This version includes gate corrections, but not obstacle shifts.
        self._gate_corrected_waypoints = self._nominal_waypoints.copy()

        # This is the final active path after gate correction and obstacle avoidance.
        self._waypoints = self._gate_corrected_waypoints.copy()

        # Tracks whether a gate has already been updated from observed pose.
        self._gate_updated = [False] * len(NOMINAL_GATE_POS)

        self._tick = 0
        self._finished = False
        self._last_yaw = 0.0

        # Build initial trajectory.
        self._rebuild_spline()

    # ------------------------------------------------------------------
    # Small utility functions
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _closest_angle(angle: float, reference: float) -> float:
        """Return angle equivalent to angle but closest to reference.

        This avoids sudden jumps from +pi to -pi.
        """
        return float(reference + ((angle - reference + np.pi) % (2.0 * np.pi) - np.pi))

    def _elapsed_time(self) -> float:
        """Current controller time in seconds."""
        return min(self._tick / self._freq, self._t_total)

    def _is_waypoint_past(self, wp_idx: int, t_now: float) -> bool:
        """Check whether a waypoint is already safely in the past."""
        if not hasattr(self, "_t_knots"):
            return False

        return self._t_knots[wp_idx] < t_now - LOCK_PAST_WAYPOINT_MARGIN_S

    # ------------------------------------------------------------------
    # Spline timing
    # ------------------------------------------------------------------

    def _make_time_knots(self, waypoints: NDArray[np.floating]) -> NDArray[np.floating]:
        """Assign timestamps to waypoints based on distance.

        The original controller spread all waypoints evenly across 15 seconds.
        That is slow and inconsistent because close segments get too much time
        and long segments get too little.

        This version gives each segment time proportional to distance.
        """
        segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)

        segment_times = np.maximum(
            segment_lengths / TARGET_SPEED_MPS,
            MIN_SEGMENT_TIME,
        )

        return np.concatenate(([0.0], np.cumsum(segment_times)))

    def _rebuild_spline(self):
        """Rebuild the position, velocity, and acceleration splines.

        This is called whenever gate corrections or obstacle avoidance modify
        the waypoint path.
        """
        t_knots = self._make_time_knots(self._waypoints)

        # Retiming loop:
        # Build a spline, check its speed/acceleration, and stretch time if needed.
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

            # Stretch all times. Velocity scales by 1/scale.
            # Acceleration scales by 1/scale^2.
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
        """Update gate-center waypoints using observed gate positions.

        This makes the path more accurate when gates are randomized.
        """
        if "gates_pos" not in obs:
            return False

        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)

        if gates_pos.ndim != 2 or gates_pos.shape[0] < len(NOMINAL_GATE_POS):
            return False

        t_now = self._elapsed_time()
        changed = False

        for gate_i, wp_i in GATE_WAYPOINT_IDX.items():
            # Do not change a gate waypoint after it has already been passed.
            if self._is_waypoint_past(wp_i, t_now):
                continue

            observed_pos = gates_pos[gate_i].copy()

            if not np.all(np.isfinite(observed_pos)):
                continue

            # Aim slightly below the observed gate center.
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
        """Push one waypoint away from obstacles in the x-y plane.

        This is intentionally simple. It does not run a full planner.
        It only makes sure the spline control points do not sit too close
        to obstacle centers.
        """
        shifted = waypoint.copy()

        for obs_pos in obs_positions:
            if not np.all(np.isfinite(obs_pos)):
                continue

            delta_xy = shifted[:2] - obs_pos[:2]
            dist_xy = np.linalg.norm(delta_xy)

            if dist_xy < OBSTACLE_CLEARANCE_RADIUS:
                if dist_xy < 1e-6:
                    # If exactly on the obstacle, push in a fixed direction.
                    direction = np.array([1.0, 0.0])
                else:
                    direction = delta_xy / dist_xy

                shifted[:2] = obs_pos[:2] + direction * OBSTACLE_CLEARANCE_RADIUS

        return shifted

    def _update_obstacle_avoidance(self, obs: dict) -> bool:
        """Recompute obstacle-safe waypoints without cumulative drift.

        Important:
        This starts from gate-corrected waypoints every time.
        It does not keep shifting already-shifted waypoints farther and farther.
        """
        if "obstacles_pos" not in obs:
            return False

        obs_positions = np.asarray(obs["obstacles_pos"], dtype=np.float64)

        if obs_positions.size == 0:
            return False

        obs_positions = obs_positions.reshape(-1, 3)

        t_now = self._elapsed_time()

        candidate_waypoints = self._gate_corrected_waypoints.copy()

        for wp_i in range(len(candidate_waypoints)):
            # Keep gate centers fixed. We want to pass through gates accurately.
            if wp_i in GATE_WAYPOINT_IDX.values():
                continue

            # Avoid changing waypoints that have already been passed.
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
        """Update gates and obstacles, then rebuild the spline if needed."""
        gates_changed = self._update_gate_waypoints(obs)
        obstacles_changed = self._update_obstacle_avoidance(obs)

        if gates_changed or obstacles_changed:
            self._rebuild_spline()

    # ------------------------------------------------------------------
    # Yaw calculation
    # ------------------------------------------------------------------

    def _compute_yaw_and_rate(
        self,
        des_vel: NDArray[np.floating],
        des_acc: NDArray[np.floating],
    ) -> tuple[float, float]:
        """Compute desired yaw and yaw-rate from trajectory direction.

        The drone points along the x-y velocity direction.
        """
        vx, vy = float(des_vel[0]), float(des_vel[1])
        ax, ay = float(des_acc[0]), float(des_acc[1])

        speed_xy_sq = vx * vx + vy * vy

        if speed_xy_sq < YAW_MIN_SPEED * YAW_MIN_SPEED:
            return self._wrap_to_pi(self._last_yaw), 0.0

        raw_yaw = float(np.arctan2(vy, vx))

        # Pick the equivalent yaw angle closest to the previous yaw.
        continuous_yaw = self._closest_angle(raw_yaw, self._last_yaw)

        # d/dt atan2(vy, vx)
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
        """Compute the desired drone state.

        Returns:
            [x, y, z,
             vx, vy, vz,
             ax, ay, az,
             yaw,
             rrate, prate, yrate]
        """
        # Update path before computing the desired state.
        self._update_path_from_observation(obs)

        t = self._elapsed_time()

        if t >= self._t_total:
            self._finished = True

        des_pos = self._spline(t)
        des_vel = self._vel_spline(t)
        des_acc = self._acc_spline(t)

        des_yaw, des_yaw_rate = self._compute_yaw_and_rate(des_vel, des_acc)

        # Roll-rate and pitch-rate are left as zero.
        # Yaw-rate is provided as feed-forward.
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
        """Advance internal time after each simulation step."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset controller state at the beginning of a new episode."""
        self._tick = 0
        self._finished = False
        self._last_yaw = 0.0

        self._gate_updated = [False] * len(NOMINAL_GATE_POS)

        self._gate_corrected_waypoints = self._nominal_waypoints.copy()
        self._waypoints = self._gate_corrected_waypoints.copy()

        self._rebuild_spline()
            
    def render_callback(self, sim: "Sim"):
        """Visualize the remaining trajectory, current setpoint, and nominal waypoints."""
        if not HAS_VIZ:
            return

        t_now = self._elapsed_time()

        if t_now < self._t_total:
            t_vals = np.linspace(t_now, self._t_total, 120)
        else:
            t_vals = np.array([self._t_total, self._t_total])

        trajectory = self._spline(t_vals)
        setpoint = self._spline(t_now).reshape(1, -1)

        # Original trajectory visualization
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        # Original current setpoint visualization
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)

        # Added: fixed nominal waypoints visualization
        draw_points(sim, NOMINAL_WAYPOINTS, rgba=(1.0, 0.5, 0.0, 1.0), size=0.04)