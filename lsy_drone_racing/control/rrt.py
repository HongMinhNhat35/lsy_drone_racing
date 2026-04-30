"""RRT (Rapidly-exploring Random Tree) 3D path planner.

Drop-in replacement for astar_3d. Returns a smoothed, collision-free path
between two 3D coordinates, avoiding a given set of obstacles.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

Coord3D = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Collision checking
# ---------------------------------------------------------------------------

def _is_clear(
    a: np.ndarray,
    b: np.ndarray,
    obstacles: list[np.ndarray],
    clearance: float,
    n_checks: int = 12,
) -> bool:
    """Return True if the segment a→b stays at least *clearance* from all obstacles."""
    for t in np.linspace(0.0, 1.0, n_checks):
        pt = a + t * (b - a)
        for obs in obstacles:
            if np.linalg.norm(pt[:2] - obs[:2]) < clearance:
                return False
    return True


# ---------------------------------------------------------------------------
# String pulling (path smoothing)
# ---------------------------------------------------------------------------

def _smooth_path(
    path: list[np.ndarray],
    obstacles: list[np.ndarray],
    clearance: float,
) -> list[np.ndarray]:
    """Remove redundant waypoints using greedy line-of-sight string pulling."""
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0

    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _is_clear(path[i], path[j], obstacles, clearance):
                break
            j -= 1
        smoothed.append(path[j])
        i = j

    return smoothed


# ---------------------------------------------------------------------------
# RRT
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("pos", "parent")

    def __init__(self, pos: np.ndarray, parent: Optional["_Node"] = None):
        self.pos = pos
        self.parent = parent


def rrt_3d(
    start: Coord3D,
    goal: Coord3D,
    obstacles: list[Coord3D],
    obstacle_clearance: float = 0.0,
    step_size: float = 0.2,
    max_iter: int = 3000,
    goal_sample_rate: float = 0.15,
    goal_reach_dist: float = 0.25,
    x_bounds: tuple[float, float] = (-2.0, 2.0),
    y_bounds: tuple[float, float] = (-2.0, 2.0),
    z_bounds: tuple[float, float] = (0.0,  2.0),
    smooth: bool = True,
    seed: Optional[int] = None,
) -> Optional[list[Coord3D]]:
    """Find a collision-free path from *start* to *goal* using RRT.

    Parameters
    ----------
    start               : (x, y, z) start position.
    goal                : (x, y, z) goal position.
    obstacles           : List of (x, y, z) obstacle positions.
    obstacle_clearance  : Minimum x-y distance to keep from obstacles.
    step_size           : Max distance the tree extends per iteration (metres).
    max_iter            : Maximum number of iterations before giving up.
    goal_sample_rate    : Probability of sampling the goal directly (0–1).
    goal_reach_dist     : Distance threshold to consider the goal reached.
    x_bounds            : Sampling bounds in x.
    y_bounds            : Sampling bounds in y.
    z_bounds            : Sampling bounds in z.
    smooth              : Apply string-pulling smoothing to the raw RRT path.
    seed                : Optional random seed for reproducibility.

    Returns
    -------
    List of (x, y, z) waypoints from start to goal, or None if no path found.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    obs_np = [np.array(o, dtype=float) for o in obstacles]
    start_np = np.array(start, dtype=float)
    goal_np = np.array(goal, dtype=float)

    # Direct path check — skip RRT if already clear
    if _is_clear(start_np, goal_np, obs_np, obstacle_clearance):
        return [tuple(start_np), tuple(goal_np)]

    root = _Node(start_np)
    nodes: list[_Node] = [root]

    for _ in range(max_iter):
        # Sample: bias toward goal occasionally
        if random.random() < goal_sample_rate:
            sample = goal_np.copy()
        else:
            sample = np.array([
                random.uniform(*x_bounds),
                random.uniform(*y_bounds),
                random.uniform(*z_bounds),
            ])

        # Nearest node
        dists = [np.linalg.norm(n.pos - sample) for n in nodes]
        nearest = nodes[int(np.argmin(dists))]

        # Steer
        direction = sample - nearest.pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            continue
        direction /= dist
        new_pos = nearest.pos + direction * min(step_size, dist)

        # Collision check
        if not _is_clear(nearest.pos, new_pos, obs_np, obstacle_clearance):
            continue

        new_node = _Node(new_pos, parent=nearest)
        nodes.append(new_node)

        # Goal reached?
        if np.linalg.norm(new_pos - goal_np) < goal_reach_dist:
            # Connect to goal
            if _is_clear(new_pos, goal_np, obs_np, obstacle_clearance):
                goal_node = _Node(goal_np, parent=new_node)
            else:
                goal_node = new_node

            # Reconstruct path
            path: list[np.ndarray] = []
            node = goal_node
            while node is not None:
                path.append(node.pos)
                node = node.parent
            path.reverse()

            if smooth:
                path = _smooth_path(path, obs_np, obstacle_clearance)

            return [tuple(float(v) for v in p) for p in path]

    return None  # No path found within max_iter
