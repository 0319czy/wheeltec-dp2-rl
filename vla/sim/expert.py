"""Expert controllers for the 12 VLA instructions (spec: specs/VLA-TASK-SPEC.md 1.3).

The expert is a *data labeller only*: it uses privileged state (robot pose + true bucket
positions), never appears in the dataset (only images + text + actions are saved).

All actions are 10 Hz physical commands (v [m/s], w [rad/s]) inside the spec action space.

Note on the turn-instr constants (3/4): the spec's nominal constants (v=0.15, w=+-0.8)
produce a forward displacement of only 0.04-0.16 m with the final low-level checkpoint,
which fails the "forward displacement > 0.15 m" criterion on most seeds (measured).
The defaults below use v=0.25, |w|=0.6 which keeps the same "curve forward while turning
left/right" semantics but passes both criteria reliably (see sim/README.md).  Pass
`tuned_turns=False` to get the literal spec constants back.

Closed-loop instructions 7-11 use the spec P controllers on the true bearing/distance.
"""
from __future__ import annotations

import math

import numpy as np

ACTION_LIMIT = 0.35
OMEGA_LIMIT = 1.0

# colour ids, same convention as env_vla.info
RED, BLUE, GREEN = 0, 1, 2


def _wrap(angle):
    """Wrap an angle to (-pi, pi]."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _clip(value, low, high):
    return max(low, min(high, value))


def _clip_omega(value):
    return _clip(float(value), -OMEGA_LIMIT, OMEGA_LIMIT)


class Expert:
    """Stateless (per step) expert for the 12 instructions.

    Parameters
    ----------
    tuned_turns : use v=0.25, |w|=0.6 for instructions 3/4 instead of the literal
                  spec constants v=0.15, w=+-0.8 (needed to satisfy the success judge).
    obstacle_gain : gain of the emergency "get away from red" term of instruction 11.
    red_clearance : red distance below which that emergency term and slowdown kick in.
    detour_offset : lateral offset (m) of the temporary waypoint around the red bucket.
    path_clearance : angular tolerance (rad) for treating red as blocking the blue direction.
    heading_gain : P gain used to steer to the aim point of instruction 11.
    """

    def __init__(self, tuned_turns=True, obstacle_gain=1.2, red_clearance=0.55,
                 avoid_clearance=0.35, forward_clearance=0.28, detour_clearance=0.55, detour_offset=0.45,
                 detour_ahead=0.30, path_clearance=0.85, heading_gain=1.7, slow_near_red=0.0):
        self.tuned_turns = bool(tuned_turns)
        self.obstacle_gain = float(obstacle_gain)
        self.red_clearance = float(red_clearance)
        self.avoid_clearance = float(avoid_clearance)
        self.forward_clearance = float(forward_clearance)
        self.detour_clearance = float(detour_clearance)
        self.detour_offset = float(detour_offset)
        self.detour_ahead = float(detour_ahead)
        self.path_clearance = float(path_clearance)
        self.heading_gain = float(heading_gain)
        self.slow_near_red = float(slow_near_red)

    # ------------------------------------------------------------------ public API
    def act(self, instr_id, info):
        """Return the 10 Hz action (v, w) for one instruction given the current info dict."""
        instr_id = int(instr_id)
        dist = info["dist"]
        heading = info["heading_err_by_color"]

        if instr_id == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        if instr_id == 1:
            obstacles = [(dist[c], heading[c], self.forward_clearance) for c in (0, 1, 2)]
            target = self._goal_with_avoidance(info, 0.0, 1.30, obstacles)
            return np.array([0.25, _clip_omega(self.heading_gain * target)], dtype=np.float32)
        if instr_id == 2:
            return np.array([-0.20, 0.0], dtype=np.float32)
        if instr_id == 3:
            w = 0.6 if self.tuned_turns else 0.8
            return np.array([0.25 if self.tuned_turns else 0.15, w], dtype=np.float32)
        if instr_id == 4:
            w = -0.6 if self.tuned_turns else -0.8
            return np.array([0.25 if self.tuned_turns else 0.15, w], dtype=np.float32)
        if instr_id == 5:
            return np.array([0.0, 1.0], dtype=np.float32)
        if instr_id == 6:
            return np.array([0.0, -1.0], dtype=np.float32)
        if instr_id == 7:
            return np.array([0.0, _clip_omega(1.5 * heading[RED])], dtype=np.float32)
        if instr_id == 8:
            return self._approach_avoid(info, RED, (BLUE, GREEN))
        if instr_id == 9:
            return self._approach_avoid(info, BLUE, (RED, GREEN))
        if instr_id == 10:
            return self._approach_avoid(info, GREEN, (RED, BLUE))
        if instr_id == 11:
            return self._avoid_red(info)
        raise ValueError("unknown instruction id: %r" % (instr_id,))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _approach(distance, heading_err):
        """Spec 1.3 instructions 8-10: drive to the target until d ~= 0.40 m."""
        if distance > 0.95:
            v = 0.32
        else:
            v = _clip(1.0 * (distance - 0.40), 0.0, 0.32)
        w = _clip_omega(1.5 * heading_err)
        return np.array([v, w], dtype=np.float32)

    def _goal_with_avoidance(self, info, goal_bearing, goal_distance, obstacles):
        """Steer toward goal_bearing, skirting any obstacle that blocks the corridor.

        obstacles: list of (distance, bearing_relative_to_heading, clearance_m).
        The tangent line at `clearance` around the closest blocking bucket is used; the
        side is chosen toward the goal bearing.  Purely a per-step function of the visible
        scene, so a BC policy can learn it.
        """
        goal_bearing = _wrap(goal_bearing)
        best = None
        for distance, bearing, clearance in obstacles:
            if distance <= 0.0 or distance > 1.15:
                continue
            delta = _wrap(bearing - goal_bearing)
            along = distance * math.cos(delta)
            lateral = distance * math.sin(delta)
            if along <= 0.05 or along > goal_distance + 0.15:
                continue
            if abs(lateral) > clearance + 0.12:
                continue
            urgency = distance - abs(lateral)
            if best is None or urgency < best[0]:
                best = (urgency, distance, bearing, clearance)
        if best is None:
            return goal_bearing
        _, distance, bearing, clearance = best
        ratio = _clip(clearance / max(distance, 0.20), 0.0, 0.85)
        alpha = math.asin(ratio)
        side = 1.0 if _wrap(goal_bearing - bearing) >= 0.0 else -1.0
        return _wrap(bearing + side * alpha)

    def _approach_avoid(self, info, target_color, other_colors):
        dist = info["dist"]
        heading = info["heading_err_by_color"]
        obstacles = [(dist[c], heading[c], self.avoid_clearance) for c in other_colors]
        target_bearing = self._goal_with_avoidance(info, heading[target_color], dist[target_color], obstacles)
        distance = dist[target_color]
        if distance > 0.95:
            v = 0.32
        else:
            v = _clip(1.0 * (distance - 0.40), 0.0, 0.32)
        w = _clip_omega(self.heading_gain * target_bearing)
        return np.array([v, w], dtype=np.float32)

    def _avoid_red(self, info):
        """Spec 1.3 instruction 11: reach blue while keeping clear of red.

        Both red (clearance red_clearance, the judge wants min d_red > 0.40) and green
        (avoid_clearance, purely against collisions) are treated as obstacles; the tangent
        steering of _goal_with_avoidance picks the blocking one and skirts it toward blue.
        """
        dist = info["dist"]
        heading = info["heading_err_by_color"]
        obstacles = [(dist[RED], heading[RED], self.red_clearance),
                     (dist[GREEN], heading[GREEN], self.avoid_clearance)]
        target_bearing = self._goal_with_avoidance(info, heading[BLUE], dist[BLUE], obstacles)
        distance = dist[BLUE]
        if distance > 0.95:
            v = 0.32
        else:
            v = _clip(1.0 * (distance - 0.40), 0.0, 0.32)
        return np.array([v, _clip_omega(self.heading_gain * target_bearing)], dtype=np.float32)


def expert_action(env, instr_id, info=None):
    """Convenience helper: one expert action for an env that already has a scene."""
    if info is None:
        raise ValueError("info is required")
    return Expert().act(instr_id, info)
