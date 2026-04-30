"""RRT based drone racing controller."""

from typing import Any

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.rrt import rrt_3d


class PIDTracker:
    """Per-axis PID that outputs a correction in that axis."""

    def __init__(
        self, kp: float, ki: float, kd: float, max_integral: float = 2.0, max_output: float = 5.0
    ):
        """Initialize the PID tracker with gains and limits."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.max_output = max_output
        self._integral = np.zeros(3)
        self._last_error = np.zeros(3)
        self._first = True

    def reset(self) -> None:
        """Reset integrator and derivative state."""
        self._integral = np.zeros(3)
        self._last_error = np.zeros(3)
        self._first = True

    def update(self, error: np.ndarray, dt: float) -> np.ndarray:
        """Compute PID output given the current error and timestep."""
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.max_integral, self.max_integral)
        if self._first:
            derivative = np.zeros(3)
            self._first = False
        else:
            derivative = (error - self._last_error) / max(dt, 1e-6)
        self._last_error = error.copy()
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return np.clip(output, -self.max_output, self.max_output)


class RRTController(Controller):
    """Drone racing controller using RRT path planning and a cubic spline trajectory."""

    def __init__(self, obs: dict, info: dict, config: Any):
        """Initialize the RRT controller and build the initial spline trajectory."""
        super().__init__(obs, info, config)

        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self.dt = 1.0 / config.env.freq

        # -------------------------------------------------------------
        # Speed / tracking parameters
        # -------------------------------------------------------------
        self.cruise_speed = 3.0
        self.look_ahead_dist = 0.15

        self.current_s = 0.0
        self._arc_length = 1.0
        self._t_total = 1.0

        # -------------------------------------------------------------
        # Planning parameters
        # -------------------------------------------------------------
        self.detour_margin = 0.22       # obstacle clearance radius (m)
        self.gate_offset = 0.21         # how far before/after gate to place waypoints
        self.max_obstacle_dist = 2.0    # only replan if obstacle is within this dist of gate

        # RRT-specific
        self.rrt_step_size = 0.5
        self.rrt_max_iter = 100
        self.rrt_goal_sample_rate = 0.5
        self.rrt_goal_reach_dist = 0.65
        self.rrt_bounds = dict(
            x_bounds=(-2.0, 2.0),
            y_bounds=(-2.0, 2.0),
            z_bounds=(0.05, 2.0),
        )

        # -------------------------------------------------------------
        # PID trackers
        # -------------------------------------------------------------
        self.pid_pos = PIDTracker(0.4, 0.0, 0.10, max_integral=1.0, max_output=1.5)
        self.pid_vel = PIDTracker(0.10, 0.0, 0.05, max_integral=0.5, max_output=2.0)
        self.pid_acc = PIDTracker(0.01, 0.0, 0.00, max_integral=0.2, max_output=1.0)

        # -------------------------------------------------------------
        # State cache for change detection
        # -------------------------------------------------------------
        self._last_gates_pos = None
        self._last_gates_quat = None
        self._last_obstacles_pos = None
        self._last_target_gate = -2

        self._gate_corners: list = []
        self._visual_trajectory: np.ndarray = np.zeros((2, 3))

        self._build_spline(obs)

    # ------------------------------------------------------------------
    # Arc-length parametrisation
    # ------------------------------------------------------------------

    def _build_arc_length_table(self, n_samples: int = 500) -> None:
        """Build a lookup table mapping arc length to spline parameter t."""
        t_samp = np.linspace(0, self._t_total, n_samples)
        pts = self.spline(t_samp)
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc_s = np.concatenate([[0.0], np.cumsum(seg_lens)])

        self._arc_length = float(arc_s[-1])
        self._s_to_t = CubicSpline(arc_s, t_samp)

    def _s_to_spline(self, s: float) -> float:
        """Convert an arc-length value to a spline parameter t."""
        s = float(np.clip(s, 0.0, self._arc_length))
        return float(np.clip(float(self._s_to_t(s)), 0.0, self._t_total))

    # ------------------------------------------------------------------
    # Gate frame virtual obstacles
    # ------------------------------------------------------------------

    def _gate_frame_obstacles(
        self, gates_pos: np.ndarray, gates_quat: np.ndarray
    ) -> list:
        """Generate virtual obstacle points around gate frames to guide RRT around them."""
        virtual_obs = []
        self._gate_corners = []
        GATE_INNER_HALF = 0.20
        GATE_OUTER_HALF = 0.40

        for i in range(len(gates_pos)):
            r = R.from_quat(gates_quat[i])
            lateral = r.apply([0, 1, 0])
            lateral = np.array([lateral[0], lateral[1], 0.0])
            lateral /= max(np.linalg.norm(lateral), 1e-6)

            for half in (GATE_INNER_HALF, GATE_OUTER_HALF):
                corners = []
                for lat_sign in (+1, -1):
                    for z_sign in (+1, -1):
                        corner = (
                            gates_pos[i]
                            + lat_sign * half * lateral
                            + np.array([0, 0, z_sign * half])
                        )
                        if half == GATE_OUTER_HALF:
                            corners.append(corner)
                        virtual_obs.append(corner)
                        self._gate_corners.append(corner)

                if half == GATE_OUTER_HALF:
                    edges = [
                        (corners[0], corners[2]),
                        (corners[1], corners[3]),
                        (corners[0], corners[1]),
                        (corners[2], corners[3]),
                    ]
                    for a, b in edges:
                        for t in (0.2, 0.4, 0.6, 0.8):
                            pt = a + t * (b - a)
                            virtual_obs.append(pt)
                            self._gate_corners.append(pt)

        return virtual_obs

    # ------------------------------------------------------------------
    # Spline builder
    # ------------------------------------------------------------------

    def _build_spline(self, obs: dict) -> None:
        """Build an RRT-guided cubic spline trajectory through all remaining gates."""
        start_pos = obs["pos"]
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]
        obstacles = obs["obstacles_pos"]

        target_gate = int(obs["target_gate"])

        remaining_pos = gates_pos[target_gate:]
        remaining_quat = gates_quat[target_gate:]

        gate_points = self._gate_frame_obstacles(gates_pos, gates_quat)

        # Sample obstacle rods at regular height intervals
        sampled_rods = []
        ROD_MAX_HEIGHT = 2.0
        ROD_STEP = 0.20
        for rod_pos in obstacles:
            for z in np.arange(0.0, ROD_MAX_HEIGHT + ROD_STEP, ROD_STEP):
                sampled_rods.append(np.array([rod_pos[0], rod_pos[1], z]))

        all_obstacles = gate_points + sampled_rods

        # Pre-compute gate approach / exit waypoints and normals
        raw_pre_gate_waypoints = []
        raw_waypoints = []
        raw_post_gate_waypoints = []
        gate_normals = []

        for i in range(len(remaining_pos)):
            rot = R.from_quat(remaining_quat[i])
            gate_normal = rot.apply([1, 0, 0])

            pre_wp  = remaining_pos[i] - gate_normal * self.gate_offset
            post_wp = remaining_pos[i] + gate_normal * self.gate_offset

            raw_pre_gate_waypoints.append(pre_wp)
            raw_waypoints.append(remaining_pos[i].copy())
            raw_post_gate_waypoints.append(post_wp)
            gate_normals.append(gate_normal)

        final_waypoints = []

        # --- start → first gate ---
        path = rrt_3d(
            start=tuple(start_pos),
            goal=tuple(raw_pre_gate_waypoints[0]),
            obstacles=all_obstacles,
            obstacle_clearance=self.detour_margin,
            step_size=self.rrt_step_size,
            max_iter=self.rrt_max_iter,
            goal_sample_rate=self.rrt_goal_sample_rate,
            goal_reach_dist=self.rrt_goal_reach_dist,
            **self.rrt_bounds,
        )
        if path is None:
            path = [tuple(start_pos), tuple(raw_pre_gate_waypoints[0])]
        for j, point in enumerate(path):
            if j % 3 == 0:
                final_waypoints.append(point)
        final_waypoints.append(tuple(raw_waypoints[0]))

        # --- gate i-1 exit → gate i ---
        for i in range(1, len(remaining_pos)):
            path = rrt_3d(
                start=tuple(raw_post_gate_waypoints[i - 1]),
                goal=tuple(raw_pre_gate_waypoints[i]),
                obstacles=all_obstacles,
                obstacle_clearance=self.detour_margin,
                step_size=self.rrt_step_size,
                max_iter=self.rrt_max_iter,
                goal_sample_rate=self.rrt_goal_sample_rate,
                goal_reach_dist=self.rrt_goal_reach_dist,
                **self.rrt_bounds,
            )
            if path is None:
                path = [
                    tuple(raw_post_gate_waypoints[i - 1]),
                    tuple(raw_pre_gate_waypoints[i]),
                ]
            for k, point in enumerate(path):
                if k % 3 == 0:
                    final_waypoints.append(point)
            final_waypoints.append(tuple(raw_waypoints[i]))

        # Ensure a clean exit after the last gate
        final_waypoints.append(tuple(raw_waypoints[-1]))
        final_waypoints.append(tuple(raw_post_gate_waypoints[-1]))

        waypoints = np.vstack(final_waypoints)
        self._t_total = len(waypoints) - 1
        t_steps = np.arange(len(waypoints))

        current_vel = np.array(obs.get("vel", [0.0, 0.0, 0.0]))
        speed = np.linalg.norm(current_vel)

        if speed > 0.1:
            dist_to_next = np.linalg.norm(waypoints[1] - waypoints[0])
            start_tangent = (current_vel / speed) * dist_to_next
            self.spline = CubicSpline(
                t_steps, waypoints, bc_type=((1, start_tangent), "natural")
            )
        else:
            self.spline = CubicSpline(t_steps, waypoints, bc_type="natural")

        self._build_arc_length_table()
        self._visual_trajectory = self.spline(np.linspace(0, self._t_total, 800))

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def _state_changed(self, obs: dict) -> bool:
        """Return True if the environment has changed enough to warrant a replan."""
        if self._last_gates_pos is None:
            return True

        target_gate = int(obs["target_gate"])

        if target_gate >= 0:
            if not np.allclose(
                obs["gates_pos"][target_gate],
                self._last_gates_pos[target_gate],
                atol=1e-2,
            ):
                return True

        if target_gate >= 0:
            target_gate_pos = obs["gates_pos"][target_gate]
            for i, obs_pos in enumerate(obs["obstacles_pos"]):
                if np.linalg.norm(target_gate_pos - obs_pos) > self.max_obstacle_dist:
                    continue
                if not np.allclose(obs_pos, self._last_obstacles_pos[i], atol=0.001):
                    return True

        return False

    def _cache_state(self, obs: dict) -> None:
        """Cache the current gate and obstacle state for change detection."""
        self._last_gates_pos = obs["gates_pos"].copy()
        self._last_gates_quat = obs["gates_quat"].copy()
        self._last_obstacles_pos = obs["obstacles_pos"].copy()
        self._last_target_gate = int(obs["target_gate"])

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Compute the control action for the current timestep."""
        if self._state_changed(obs):
            actual_pos_before = np.array(obs["pos"])
            try:
                self._build_spline(obs)
                self._cache_state(obs)

                # Re-locate the drone on the new spline
                s_search = np.linspace(
                    0.0, min(self._arc_length, self.cruise_speed * 3.0), 200
                )
                t_search = np.array([self._s_to_spline(s) for s in s_search])
                pts = self.spline(t_search)
                dists = np.linalg.norm(pts - actual_pos_before, axis=1)
                self.current_s = float(s_search[np.argmin(dists)])

            except Exception:  # noqa: BLE001
                pass

        target_s = min(self.current_s + self.look_ahead_dist, self._arc_length)
        t = self._s_to_spline(target_s)

        ref_pos = self.spline(t)
        spline_tangent = self.spline(t, 1)
        ds_dt = max(np.linalg.norm(spline_tangent), 1e-6)
        dt_ds = 1.0 / ds_dt

        # Curvature-based speed scaling — slow down on tight turns
        curvature = np.linalg.norm(self.spline(t, 2))
        speed_scale = 1.0 / (1.0 + 2.0 * curvature)
        effective_speed = self.cruise_speed * speed_scale

        ref_vel = spline_tangent * dt_ds * effective_speed
        ref_acc = self.spline(t, 2) * dt_ds**2 * effective_speed**2

        actual_pos = np.array(obs["pos"])
        actual_vel = np.array(obs["vel"])

        pos_correction = self.pid_pos.update(ref_pos - actual_pos, self.dt)
        vel_correction = self.pid_vel.update(ref_vel - actual_vel, self.dt)
        acc_correction = self.pid_acc.update(ref_acc, self.dt)

        return np.array(
            [
                *(ref_pos + pos_correction),
                *(ref_vel + vel_correction),
                *(ref_acc + acc_correction),
                0, 0, 0, 0,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Update arc-length progress and check for episode completion."""
        drone_pos = np.array(obs["pos"])

        s_search = np.linspace(
            max(0.0, self.current_s - 0.2),
            min(self._arc_length, self.current_s + self.cruise_speed * 0.5),
            60,
        )
        t_search = np.array([self._s_to_spline(s) for s in s_search])
        pts = self.spline(t_search)
        dists = np.linalg.norm(pts - drone_pos, axis=1)
        best_s = float(s_search[np.argmin(dists)])

        time_advance = self.cruise_speed * self.dt
        self.current_s = float(
            np.clip(max(self.current_s + time_advance * 0.5, best_s), 0.0, self._arc_length)
        )

        if int(obs["target_gate"]) == -1:
            self._finished = True

        self._tick += 1
        return self._finished

    def episode_reset(self) -> None:
        """Reset controller state at the start of a new episode."""
        self._tick = 0
        self._finished = False
        self.current_s = 0.0
        self.pid_pos.reset()
        self.pid_vel.reset()
        self.pid_acc.reset()
        self._last_gates_pos = None
        self._last_gates_quat = None
        self._last_obstacles_pos = None
        self._last_target_gate = -2

    def render_callback(self, sim: Any) -> None:
        """Draw the planned trajectory and current setpoint in the visualiser."""
        draw_line(sim, self._visual_trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

        t_now = self._s_to_spline(min(self.current_s + self.look_ahead_dist, self._arc_length))
        setpoint = self.spline(t_now).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

        if self._gate_corners:
            corners = np.array(self._gate_corners)
            draw_points(sim, corners, rgba=(1.0, 0.5, 0.0, 1.0), size=0.03)
