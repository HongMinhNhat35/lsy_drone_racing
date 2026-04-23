"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray
try:
    from crazyflow.sim.visualize import draw_line, draw_points
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
NOMINAL_GATE_POS = np.array([
    [0.5,  0.25, 0.7 ],
    [1.05, 0.75, 1.2 ],
    [-1.0,-0.25, 0.7 ],
    [0.0, -0.75, 1.2 ],
])
OBSTACLE_RADIUS = 0.15  # meters of clearance around each obstacle (actual is 0.015m, padding for safety)
OBSTACLE_SHIFT  = 0.21  # how far to push the waypoint sideways if too close
GATE_Z_OFFSET = -0.1

class StateController(Controller):
    """State controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # Same waypoints as in the attitude controller. Determined by trial and error.
        self._waypoints = np.array([
            [-1.5,  0.75, 0.05],  # 0 start
            [-1.0,  0.55, 0.4 ],  # 1
            [ 0.0,  0.45, 0.7 ],  # 2 approach gate 0
            [ 0.5,  0.25, 0.7 ],  # 3 ← gate 0 center
            [ 1.3, -0.15, 0.9 ],  # 4 approach gate 1
            [ 1.05, 0.75, 1.2 ],  # 5 ← gate 1 center
            [ 0.65, 1.0, 1.2 ],  # 6
            [-0.2, -0.05, 0.6 ],  # 7
            [-0.6, -0.2,  0.6 ],  # 8 approach gate 2
            [-1.0, -0.25, 0.7 ],  # 9 ← gate 2 center
            [-1.5, -0.4, 0.7 ], #10
            [-1.5, -0.5,  1.2 ],  # 11
            [-1.0, -0.7,  1.2 ],  # 12 approach gate 3
            [-0.5, -0.65,  1.2 ],
            [-0.2, -0.65,  1.2 ],
            [ 0.0, -0.75, 1.2 ],  # 12 ← gate 3 center
            [ 0.5, -0.75, 1.2 ],  # 13 end
        ])
        self._last_waypoints = self._waypoints.copy()
        self._gate_waypoint_idx = {0: 3, 1: 5, 2: 9, 3: 15}
        self._gate_updated = [False] * 4



        self._t_total = 15.0
        self._tick = 0
        self._finished = False
        self._rebuild_spline()


    def _shift_waypoint_from_obstacles(self, waypoint: NDArray, obs_positions: NDArray) -> NDArray:
        shifted = waypoint.copy()

        for obs_pos in obs_positions:
            delta_xy = shifted[:2] - obs_pos[:2]
            dist_xy  = np.linalg.norm(delta_xy)
            if dist_xy < OBSTACLE_RADIUS:
                if dist_xy < 1e-6:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = delta_xy / dist_xy
                shifted[:2] = obs_pos[:2] + direction * (OBSTACLE_RADIUS + OBSTACLE_SHIFT)
        return shifted

    def _update_obstacle_avoidance(self, obs: dict):
        obs_positions = obs["obstacles_pos"]
        changed = False
        for i, wp in enumerate(self._waypoints):
            if i in self._gate_waypoint_idx.values():
                continue
            shifted = self._shift_waypoint_from_obstacles(wp, obs_positions)
            if not np.allclose(shifted, wp):
                self._waypoints[i] = shifted
                changed = True
        if changed and not np.allclose(self._waypoints, self._last_waypoints):
            self._last_waypoints = self._waypoints.copy()
            self._rebuild_spline()
    def _rebuild_spline(self):
        t = np.linspace(0, self._t_total, len(self._waypoints))
        self._spline = CubicSpline(t, self._waypoints)

    def _check_and_update_gates(self, obs):
        drone_pos = obs["pos"]
        gates_pos = obs["gates_pos"]
        changed = False

        for gate_i, wp_i in self._gate_waypoint_idx.items():
            if self._gate_updated[gate_i]:
                continue
            dist_to_nominal = np.linalg.norm(drone_pos - NOMINAL_GATE_POS[gate_i])
            if dist_to_nominal < 0.7:
                observed_pos = gates_pos[gate_i].copy()
                observed_pos[2] += GATE_Z_OFFSET
                shift = np.linalg.norm(observed_pos - NOMINAL_GATE_POS[gate_i])
                if shift > 0.01:
                    self._waypoints[wp_i] = observed_pos
                    changed = True
                self._gate_updated[gate_i] = True

        if changed:
            self._rebuild_spline()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        self._check_and_update_gates(obs)
        self._update_obstacle_avoidance(obs)
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        des_pos = self._spline(t)
        des_vel = self._spline.derivative()(t)

        return np.concatenate((des_pos, des_vel, np.zeros(7)), dtype=np.float32)


    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False
        self._gate_updated = [False] * 4
        # Reset gate center waypoints to nominal
        for gate_i, wp_i in self._gate_waypoint_idx.items():
            self._waypoints[wp_i] = NOMINAL_GATE_POS[gate_i].copy()
        self._rebuild_spline()

    def render_callback(self, sim: "Sim"):
        if not HAS_VIZ:
            return
        t_now = self._tick / self._freq
        t_vals = np.linspace(t_now, self._t_total, 120)
        draw_line(sim, self._spline(t_vals), rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(sim, self._spline(t_now).reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)
