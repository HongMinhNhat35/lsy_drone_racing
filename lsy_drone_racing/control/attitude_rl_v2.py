"""RL attitude controller with gate-aware trajectory generation.

This controller loads the PPO policy trained in train_rl.py, but instead of
following a purely hard-coded spline, it builds a race-aware reference trajectory:

1. Use observed gate positions.
2. Add approach, center, and exit waypoints for each gate.
3. Shift intermediate waypoints away from obstacles.
4. Retiming uses segment distance and curvature.
5. Feed local future samples from this spline into the PPO policy.

The PPO policy is still only the low-level tracker. The trajectory generator
below is the planner.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.train_rl import Agent

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================
# RL observation settings
# These must match train_rl.py.
# ============================================================

N_OBS = 2
N_SAMPLES = 10
SAMPLES_DT = 0.1


# ============================================================
# Trajectory generation settings
# ============================================================

# Start conservative. Lower this once the controller finishes reliably.
# 15.0 is closer to the original training distribution.
# 10.5 is a faster attempt.
TARGET_TOTAL_TIME = 10.5

MIN_TOTAL_TIME = 8.8
MAX_TOTAL_TIME = 15.0

GATE_Z_OFFSET = -0.10

APPROACH_DIST = 0.45
EXIT_DIST = 0.35

POINT_OBSTACLE_CLEARANCE = 0.36
SEGMENT_OBSTACLE_CLEARANCE = 0.42
OBSTACLE_Z_LIFT = 0.18

NOMINAL_SPEED = 1.9
CURVATURE_TIME_GAIN = 0.35
GATE_TIME_GAIN = 0.18

MAX_ROLL_PITCH_RAD = 0.85

YAW_FIXED = 0.0


# Fallback gate centers if obs["gates_pos"] is unavailable.
FALLBACK_GATE_POS = np.array(
    [
        [0.5, 0.25, 0.70],
        [1.05, 0.75, 1.20],
        [-1.0, -0.25, 0.70],
        [0.0, -0.75, 1.20],
    ],
    dtype=np.float64,
)

FALLBACK_START_POS = np.array([-1.5, 0.75, 0.05], dtype=np.float64)


class AttitudeRL(Controller):
    """Deploy a trained PPO policy using a gate-aware trajectory planner."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self.freq = int(config.env.freq)

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = float(drone_params["mass"])
        self.thrust_min = float(drone_params["thrust_min"] * 4.0)
        self.thrust_max = float(drone_params["thrust_max"] * 4.0)

        self.n_obs = N_OBS
        self.n_samples = N_SAMPLES
        self.samples_dt = SAMPLES_DT

        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt,
            dtype=int,
        )

        self.basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        self.basic_obs_dim = 13

        self.expected_obs_dim = (
            self.basic_obs_dim
            + 3 * self.n_samples
            + self.n_obs * self.basic_obs_dim
            + 4
        )

        self._tick = 0
        self._finished = False
        self._needs_rebuild = True

        self._t_total = TARGET_TOTAL_TIME
        self._waypoints = None
        self._gate_center_flags = None
        self.trajectory = None

        self.agent = self._load_agent()
        self.agent.eval()

        self.prev_obs = np.zeros((self.n_obs, self.basic_obs_dim), dtype=np.float32)
        self.last_action = np.array(
            [0.0, 0.0, 0.0, self._hover_thrust_normalized()],
            dtype=np.float32,
        )

        self._build_trajectory(obs)
        self._reset_history_from_obs(obs)
        self._needs_rebuild = False

    # ============================================================
    # Agent loading
    # ============================================================

    def _load_agent(self) -> Agent:
        """Load PPO checkpoint."""
        agent = Agent((self.expected_obs_dim,), (4,)).to("cpu")

        model_path = Path(__file__).parent / "ppo_drone_racing.ckpt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Could not find PPO checkpoint at {model_path}. "
                "Train the policy first or copy ppo_drone_racing.ckpt into this folder."
            )

        try:
            state_dict = torch.load(
                model_path,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))

        agent.load_state_dict(state_dict)
        return agent

    # ============================================================
    # Trajectory planner
    # ============================================================

    def _build_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Build a gate-centered, obstacle-aware, retimed trajectory."""
        start_pos = self._get_start_pos(obs)
        gates_pos = self._get_gate_positions(obs)
        gates_quat = self._get_gate_quats(obs)
        obstacles = self._get_obstacles(obs)

        waypoints, gate_center_flags = self._build_gate_waypoints(
            start_pos=start_pos,
            gates_pos=gates_pos,
            gates_quat=gates_quat,
        )

        waypoints = self._shift_intermediate_waypoints(
            waypoints=waypoints,
            gate_center_flags=gate_center_flags,
            obstacles=obstacles,
        )

        waypoints, gate_center_flags = self._insert_obstacle_detours(
            waypoints=waypoints,
            gate_center_flags=gate_center_flags,
            obstacles=obstacles,
        )

        t_knots, t_total = self._retime_waypoints(
            waypoints=waypoints,
            gate_center_flags=gate_center_flags,
        )

        n_steps = max(int(np.ceil(t_total * self.freq)), 2)
        ts = np.linspace(0.0, t_total, n_steps)

        spline = CubicSpline(t_knots, waypoints, bc_type="clamped")

        self._t_total = float(t_total)
        self._waypoints = waypoints
        self._gate_center_flags = gate_center_flags
        self.trajectory = spline(ts).astype(np.float32)

    def _get_start_pos(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Use actual drone position as the first waypoint."""
        if "pos" not in obs:
            return FALLBACK_START_POS.copy()

        pos = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)
        if pos.shape[0] != 3 or not np.all(np.isfinite(pos)):
            return FALLBACK_START_POS.copy()

        return pos.copy()

    def _get_gate_positions(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Read gate positions from observation, or use fallback."""
        if "gates_pos" not in obs:
            return FALLBACK_GATE_POS.copy()

        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)

        if gates_pos.ndim != 2 or gates_pos.shape[1] != 3 or gates_pos.shape[0] < 4:
            return FALLBACK_GATE_POS.copy()

        gates_pos = gates_pos[:4].copy()

        if not np.all(np.isfinite(gates_pos)):
            return FALLBACK_GATE_POS.copy()

        gates_pos[:, 2] += GATE_Z_OFFSET
        return gates_pos

    def _get_gate_quats(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray | None:
        """Read gate orientations if available."""
        for key in ["gates_quat", "gate_quat", "gates_quats"]:
            if key in obs:
                quats = np.asarray(obs[key], dtype=np.float64)
                if quats.ndim == 2 and quats.shape[1] == 4 and quats.shape[0] >= 4:
                    if np.all(np.isfinite(quats[:4])):
                        return quats[:4].copy()

        return None

    def _get_obstacles(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Read obstacle positions if available."""
        if "obstacles_pos" not in obs:
            return np.zeros((0, 3), dtype=np.float64)

        obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float64)

        if obstacles.size == 0:
            return np.zeros((0, 3), dtype=np.float64)

        obstacles = obstacles.reshape(-1, 3)

        if not np.all(np.isfinite(obstacles)):
            obstacles = obstacles[np.all(np.isfinite(obstacles), axis=1)]

        return obstacles

    def _build_gate_waypoints(
        self,
        start_pos: np.ndarray,
        gates_pos: np.ndarray,
        gates_quat: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create start, approach, gate-center, exit, and final waypoints."""
        waypoints = [start_pos.astype(np.float64)]
        gate_center_flags = [False]

        centers = gates_pos.astype(np.float64)

        for i, center in enumerate(centers):
            direction = self._gate_direction(
                gate_index=i,
                gates_pos=centers,
                gates_quat=gates_quat,
                start_pos=start_pos,
            )

            approach = center - APPROACH_DIST * direction
            exit_point = center + EXIT_DIST * direction

            # Approach and exit are allowed to move for obstacle avoidance.
            waypoints.append(approach)
            gate_center_flags.append(False)

            # Gate center should not be shifted by obstacle logic.
            waypoints.append(center)
            gate_center_flags.append(True)

            waypoints.append(exit_point)
            gate_center_flags.append(False)

        # Add a final point after the last gate.
        if len(centers) >= 2:
            final_dir = self._safe_unit(centers[-1] - centers[-2])
        else:
            final_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        final_point = centers[-1] + 0.65 * final_dir
        waypoints.append(final_point)
        gate_center_flags.append(False)

        waypoints = np.asarray(waypoints, dtype=np.float64)
        gate_center_flags = np.asarray(gate_center_flags, dtype=bool)

        return waypoints, gate_center_flags

    def _gate_direction(
        self,
        gate_index: int,
        gates_pos: np.ndarray,
        gates_quat: np.ndarray | None,
        start_pos: np.ndarray,
    ) -> np.ndarray:
        """Estimate the direction through a gate.

        If gate orientation is available, use it and choose the sign that points
        from the previous target toward the next target. Otherwise use the
        direction from previous gate/start to next gate.
        """
        center = gates_pos[gate_index]

        if gate_index == 0:
            prev_point = start_pos
        else:
            prev_point = gates_pos[gate_index - 1]

        if gate_index < len(gates_pos) - 1:
            next_point = gates_pos[gate_index + 1]
        else:
            next_point = center + self._safe_unit(center - prev_point)

        desired_dir = self._safe_unit(next_point - prev_point)

        if gates_quat is not None:
            try:
                rot = R.from_quat(gates_quat[gate_index])
                # Convention can vary. x-axis is a reasonable normal candidate.
                normal = rot.as_matrix()[:, 0]
                normal = self._safe_unit(normal)

                if np.dot(normal, desired_dir) < 0.0:
                    normal = -normal

                return normal
            except ValueError:
                pass

        return desired_dir

    def _shift_intermediate_waypoints(
        self,
        waypoints: np.ndarray,
        gate_center_flags: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        """Push non-gate waypoints away from obstacle centers in x-y."""
        if obstacles.shape[0] == 0:
            return waypoints

        shifted = waypoints.copy()

        for i in range(len(shifted)):
            if gate_center_flags[i]:
                continue

            p = shifted[i].copy()

            for obs_pos in obstacles:
                delta_xy = p[:2] - obs_pos[:2]
                dist_xy = float(np.linalg.norm(delta_xy))

                if dist_xy < POINT_OBSTACLE_CLEARANCE:
                    if dist_xy < 1e-6:
                        direction = np.array([1.0, 0.0], dtype=np.float64)
                    else:
                        direction = delta_xy / dist_xy

                    p[:2] = obs_pos[:2] + direction * POINT_OBSTACLE_CLEARANCE
                    p[2] = max(p[2], obs_pos[2] + OBSTACLE_Z_LIFT)

            shifted[i] = p

        return shifted

    def _insert_obstacle_detours(
        self,
        waypoints: np.ndarray,
        gate_center_flags: np.ndarray,
        obstacles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Insert extra detour waypoints when a segment passes too close to an obstacle."""
        if obstacles.shape[0] == 0 or len(waypoints) < 2:
            return waypoints, gate_center_flags

        new_points = [waypoints[0]]
        new_flags = [gate_center_flags[0]]

        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]

            detour = self._segment_detour_point(p0, p1, obstacles)

            if detour is not None:
                new_points.append(detour)
                new_flags.append(False)

            new_points.append(p1)
            new_flags.append(gate_center_flags[i + 1])

        return np.asarray(new_points, dtype=np.float64), np.asarray(new_flags, dtype=bool)

    def _segment_detour_point(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray | None:
        """Return a detour point if segment p0-p1 is too close to any obstacle."""
        seg_xy = p1[:2] - p0[:2]
        seg_len2 = float(np.dot(seg_xy, seg_xy))

        if seg_len2 < 1e-9:
            return None

        worst_obs = None
        worst_proj = None
        worst_dist = 1e9

        for obs_pos in obstacles:
            u = float(np.clip(np.dot(obs_pos[:2] - p0[:2], seg_xy) / seg_len2, 0.0, 1.0))
            proj_xy = p0[:2] + u * seg_xy
            dist = float(np.linalg.norm(proj_xy - obs_pos[:2]))

            if dist < SEGMENT_OBSTACLE_CLEARANCE and dist < worst_dist:
                z_proj = p0[2] + u * (p1[2] - p0[2])
                worst_obs = obs_pos
                worst_proj = np.array([proj_xy[0], proj_xy[1], z_proj], dtype=np.float64)
                worst_dist = dist

        if worst_obs is None or worst_proj is None:
            return None

        away = worst_proj[:2] - worst_obs[:2]
        away_norm = float(np.linalg.norm(away))

        if away_norm < 1e-6:
            # Use a perpendicular direction to the segment.
            seg_unit = self._safe_unit(np.array([seg_xy[0], seg_xy[1], 0.0]))
            away = np.array([-seg_unit[1], seg_unit[0]], dtype=np.float64)
        else:
            away = away / away_norm

        detour = worst_proj.copy()
        detour[:2] = worst_obs[:2] + away * SEGMENT_OBSTACLE_CLEARANCE
        detour[2] = max(detour[2], worst_obs[2] + OBSTACLE_Z_LIFT)

        return detour

    def _retime_waypoints(
        self,
        waypoints: np.ndarray,
        gate_center_flags: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Create nonuniform time knots based on distance and curvature."""
        diffs = np.diff(waypoints, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        seg_lengths = np.maximum(seg_lengths, 1e-4)

        # Base time from distance.
        seg_times = seg_lengths / max(NOMINAL_SPEED, 1e-3)

        # Add time near sharp turns.
        curvature_extra = np.zeros(len(waypoints), dtype=np.float64)

        for i in range(1, len(waypoints) - 1):
            v1 = self._safe_unit(waypoints[i] - waypoints[i - 1])
            v2 = self._safe_unit(waypoints[i + 1] - waypoints[i])
            dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
            angle = float(np.arccos(dot))

            curvature_extra[i] += CURVATURE_TIME_GAIN * (angle / np.pi)

        # Add small time near gate centers to improve accuracy through gates.
        gate_extra = gate_center_flags.astype(np.float64) * GATE_TIME_GAIN

        point_extra = curvature_extra + gate_extra
        seg_times += 0.5 * point_extra[:-1] + 0.5 * point_extra[1:]

        # Preserve relative distance/curvature timing, then scale to target duration.
        raw_total = float(np.sum(seg_times))
        target_total = float(np.clip(TARGET_TOTAL_TIME, MIN_TOTAL_TIME, MAX_TOTAL_TIME))

        if raw_total < 1e-6:
            seg_times = np.ones_like(seg_times) * (target_total / len(seg_times))
        else:
            seg_times = seg_times / raw_total * target_total

        t_knots = np.concatenate(([0.0], np.cumsum(seg_times)))

        # CubicSpline requires strictly increasing knots.
        for i in range(1, len(t_knots)):
            if t_knots[i] <= t_knots[i - 1]:
                t_knots[i] = t_knots[i - 1] + 1e-3

        return t_knots, float(t_knots[-1])

    @staticmethod
    def _safe_unit(v: np.ndarray) -> np.ndarray:
        """Normalize vector with fallback."""
        v = np.asarray(v, dtype=np.float64).reshape(-1)

        if v.shape[0] == 2:
            n = float(np.linalg.norm(v))
            if n < 1e-8:
                return np.array([1.0, 0.0], dtype=np.float64)
            return v / n

        if v.shape[0] != 3:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        n = float(np.linalg.norm(v))
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        return v / n

    # ============================================================
    # RL observation construction
    # ============================================================

    def _reset_history_from_obs(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Reset previous observations and previous action."""
        basic_obs = self._basic_obs(obs)

        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1)).astype(np.float32)

        self.last_action = np.array(
            [0.0, 0.0, 0.0, self._hover_thrust_normalized()],
            dtype=np.float32,
        )

    def _basic_obs(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Return [pos, quat, vel, ang_vel] as one flat vector."""
        values = []

        for key in self.basic_obs_key:
            if key not in obs:
                raise KeyError(f"Missing observation key {key!r} required by AttitudeRL.")

            arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            values.append(arr)

        basic = np.concatenate(values, axis=0).astype(np.float32)

        if basic.shape[0] != self.basic_obs_dim:
            raise ValueError(
                f"Expected basic observation dimension {self.basic_obs_dim}, "
                f"got {basic.shape[0]}."
            )

        return basic

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> np.ndarray:
        """Build the exact flat input vector expected by train_rl.Agent."""
        basic_obs = self._basic_obs(obs)

        idx = np.clip(
            self._tick + self.sample_offsets,
            0,
            self.trajectory.shape[0] - 1,
        )

        pos = np.asarray(obs["pos"], dtype=np.float32).reshape(3)
        local_samples = (self.trajectory[idx] - pos).reshape(-1).astype(np.float32)

        obs_rl = np.concatenate(
            [
                basic_obs,
                local_samples,
                self.prev_obs.reshape(-1).astype(np.float32),
                self.last_action.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        if obs_rl.shape[0] != self.expected_obs_dim:
            raise ValueError(
                f"AttitudeRL observation has dimension {obs_rl.shape[0]}, "
                f"but expected {self.expected_obs_dim}. "
                "Check N_OBS, N_SAMPLES, and train_rl wrapper order."
            )

        self.prev_obs = np.concatenate(
            [self.prev_obs[1:, :], basic_obs[None, :]],
            axis=0,
        ).astype(np.float32)

        return obs_rl

    # ============================================================
    # Action scaling
    # ============================================================

    def _hover_thrust_normalized(self) -> float:
        """Approximate hover thrust in normalized action coordinates."""
        hover_thrust = self.drone_mass * 9.81

        mean = 0.5 * (self.thrust_max + self.thrust_min)
        scale = 0.5 * (self.thrust_max - self.thrust_min)

        if scale <= 1e-8:
            return 0.0

        return float(np.clip((hover_thrust - mean) / scale, -1.0, 1.0))

    def _scale_actions(self, actions: np.ndarray) -> np.ndarray:
        """Map normalized PPO actions to [roll, pitch, yaw, collective thrust]."""
        actions = np.asarray(actions, dtype=np.float32).reshape(4)
        actions = np.clip(actions, -1.0, 1.0)

        scale = np.array(
            [
                np.pi / 2.0,
                np.pi / 2.0,
                np.pi / 2.0,
                0.5 * (self.thrust_max - self.thrust_min),
            ],
            dtype=np.float32,
        )

        mean = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.5 * (self.thrust_max + self.thrust_min),
            ],
            dtype=np.float32,
        )

        scaled = actions * scale + mean

        scaled[0] = np.clip(scaled[0], -MAX_ROLL_PITCH_RAD, MAX_ROLL_PITCH_RAD)
        scaled[1] = np.clip(scaled[1], -MAX_ROLL_PITCH_RAD, MAX_ROLL_PITCH_RAD)
        scaled[2] = YAW_FIXED
        scaled[3] = np.clip(scaled[3], self.thrust_min, self.thrust_max)

        return scaled.astype(np.float32)

    # ============================================================
    # Main controller API
    # ============================================================

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None,
    ) -> NDArray[np.floating]:
        """Return [roll_des, pitch_des, yaw_des, collective_thrust_des]."""
        if self._needs_rebuild:
            self._build_trajectory(obs)
            self._reset_history_from_obs(obs)
            self._needs_rebuild = False

        if self._tick >= self.trajectory.shape[0] - 1:
            self._finished = True

        obs_rl = self._obs_rl(obs)
        obs_tensor = torch.tensor(obs_rl, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_tensor, _, _, _ = self.agent.get_action_and_value(
                obs_tensor,
                deterministic=True,
            )

        raw_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        # Match training: last_action stores the raw normalized policy action.
        # The actually applied yaw is then forced to zero.
        self.last_action = raw_action.copy()

        action_to_apply = raw_action.copy()
        action_to_apply[2] = 0.0

        return self._scale_actions(action_to_apply)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance internal time."""
        self._tick += 1
        return self._finished

    def episode_callback(self) -> None:
        """Reset state between episodes.

        There is no obs argument here, so the trajectory/history are rebuilt lazily
        on the next compute_control call.
        """
        self._tick = 0
        self._finished = False
        self._needs_rebuild = True