"""Safe closed-loop state controller.

Defines exactly one controller class:
    StateController

Returned command format:
[x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]

This version is intentionally conservative so that the drone should not crash
during the first second. It uses the current state to generate nearby local
position setpoints instead of commanding far-away gates directly.
"""

from __future__ import annotations

import numpy as np

from lsy_drone_racing.control.controller import Controller

try:
    from crazyflow.sim.visualize import draw_line, draw_points

    HAS_VIZ = True
except ImportError:
    draw_line = None
    draw_points = None
    HAS_VIZ = False


NOMINAL_GATE_POS = np.array(
    [[0.50, 0.25, 0.70], [1.05, 0.75, 1.20], [-1.00, -0.25, 0.70], [0.00, -0.75, 1.20]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------
# Very safe tuning
# ---------------------------------------------------------------------

TAKEOFF_TIME = 2.0
TAKEOFF_Z = 0.55

# How far ahead of the current drone position the local target can be.
# This is deliberately small to avoid huge position errors.
LOOKAHEAD_DIST = 0.18

# Desired speed while moving.
MOVE_SPEED = 0.75

# Slower speed near waypoints/gates.
NEAR_SPEED = 0.40

# Distance needed to consider a waypoint reached.
WAYPOINT_RADIUS = 0.18

# End condition.
END_RADIUS = 0.18

# Aim a bit below gate center. Set to 0.0 if it misses vertically.
GATE_Z_OFFSET = -0.02

# Add approach/exit points so it does not cut straight through sharp corners.
APPROACH_DIST = 0.25
EXIT_DIST = 0.20

FINAL_EXIT_VECTOR = np.array([0.35, 0.0, 0.0], dtype=np.float64)

YAW_MIN_SPEED = 0.05


class StateController(Controller):
    """Safe state controller using current position feedback."""

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)

        self._freq = float(config.env.freq)
        self._tick = 0
        self._finished = False

        self._start_pos = self._get_pos(obs)
        self._last_yaw = 0.0

        self._path = self._build_path(obs)
        self._wp_idx = 1

        self._current_target = self._path[0].copy()

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pos(obs: dict) -> np.ndarray:
        if "pos" in obs:
            return np.asarray(obs["pos"], dtype=np.float64).reshape(3)
        return np.zeros(3, dtype=np.float64)

    @staticmethod
    def _get_vel(obs: dict) -> np.ndarray:
        if "vel" in obs:
            return np.asarray(obs["vel"], dtype=np.float64).reshape(3)
        return np.zeros(3, dtype=np.float64)

    def _get_gate_positions(self, obs: dict) -> np.ndarray:
        if "gates_pos" in obs:
            gates = np.asarray(obs["gates_pos"], dtype=np.float64)

            if gates.ndim == 2 and gates.shape[0] >= 4 and gates.shape[1] >= 3:
                gates = gates[:4, :3].copy()

                if np.all(np.isfinite(gates)):
                    gates[:, 2] += GATE_Z_OFFSET
                    return gates

        gates = NOMINAL_GATE_POS.copy()
        gates[:, 2] += GATE_Z_OFFSET
        return gates

    # ------------------------------------------------------------------
    # Path
    # ------------------------------------------------------------------

    def _build_path(self, obs: dict) -> np.ndarray:
        """Build a safe waypoint path."""
        gates = self._get_gate_positions(obs)

        path = []

        # Start at current position.
        path.append(self._start_pos.copy())

        # Takeoff point above start.
        takeoff = self._start_pos.copy()
        takeoff[2] = TAKEOFF_Z
        path.append(takeoff)

        previous = takeoff

        for gate in gates:
            direction = gate - previous
            norm = float(np.linalg.norm(direction))

            if norm < 1e-9:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                direction = direction / norm

            approach = gate - direction * APPROACH_DIST
            exit_point = gate + direction * EXIT_DIST

            path.append(approach)
            path.append(gate.copy())
            path.append(exit_point)

            previous = gate

        path.append(gates[-1] + FINAL_EXIT_VECTOR)

        return np.asarray(path, dtype=np.float64)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _time(self) -> float:
        return self._tick / self._freq

    def _advance_waypoint_if_needed(self, pos: np.ndarray):
        """Advance to the next waypoint when close enough."""
        if self._wp_idx >= len(self._path):
            self._finished = True
            return

        target = self._path[self._wp_idx]
        dist = float(np.linalg.norm(pos - target))

        if self._wp_idx == len(self._path) - 1:
            if dist < END_RADIUS:
                self._finished = True
            return

        if dist < WAYPOINT_RADIUS:
            self._wp_idx += 1

    def _local_target_toward(
        self, pos: np.ndarray, target: np.ndarray, lookahead: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return a nearby local setpoint toward the active waypoint."""
        delta = target - pos
        dist = float(np.linalg.norm(delta))

        if dist < 1e-9:
            direction = np.zeros(3, dtype=np.float64)
            return target.copy(), direction, 0.0

        direction = delta / dist

        step = min(lookahead, dist)
        local_target = pos + direction * step

        return local_target, direction, dist

    def _desired_yaw(self, direction: np.ndarray) -> float:
        vx = float(direction[0])
        vy = float(direction[1])

        speed_xy = float(np.hypot(vx, vy))

        if speed_xy < YAW_MIN_SPEED:
            return self._last_yaw

        yaw = float(np.arctan2(vy, vx))
        self._last_yaw = self._wrap_to_pi(yaw)

        return self._last_yaw

    # ------------------------------------------------------------------
    # Main controller API
    # ------------------------------------------------------------------

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        pos = self._get_pos(obs)

        t = self._time()

        # First phase: slow vertical takeoff while holding x-y.
        if t < TAKEOFF_TIME:
            alpha = t / TAKEOFF_TIME

            target_pos = self._start_pos.copy()
            target_pos[2] = (1.0 - alpha) * self._start_pos[2] + alpha * TAKEOFF_Z

            des_vel = np.zeros(3, dtype=np.float64)
            des_vel[2] = (TAKEOFF_Z - self._start_pos[2]) / TAKEOFF_TIME

            des_acc = np.zeros(3, dtype=np.float64)
            des_yaw = 0.0

            self._current_target = target_pos.copy()

        else:
            self._advance_waypoint_if_needed(pos)

            if self._wp_idx >= len(self._path):
                self._finished = True
                target_pos = pos.copy()
                des_vel = np.zeros(3, dtype=np.float64)
                des_acc = np.zeros(3, dtype=np.float64)
                des_yaw = self._last_yaw
            else:
                active_wp = self._path[self._wp_idx]

                target_pos, direction, dist_to_wp = self._local_target_toward(
                    pos, active_wp, LOOKAHEAD_DIST
                )

                # Slow down near waypoints.
                if dist_to_wp < 0.45:
                    speed = NEAR_SPEED
                else:
                    speed = MOVE_SPEED

                des_vel = direction * speed

                # Extremely important:
                # Zero acceleration feed-forward. This avoids aggressive tilting.
                des_acc = np.zeros(3, dtype=np.float64)

                des_yaw = self._desired_yaw(direction)

                self._current_target = target_pos.copy()

        action = np.concatenate(
            (target_pos, des_vel, des_acc, np.array([des_yaw, 0.0, 0.0, 0.0], dtype=np.float64))
        ).astype(np.float32)

        return action

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
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
        self._wp_idx = 1
        self._last_yaw = 0.0

    def render_callback(self, sim):
        if not HAS_VIZ:
            return

        draw_line(sim, self._path, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._current_target.reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.03)
