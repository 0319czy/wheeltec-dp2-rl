"""VLA simulation environment for the DP2 balance car (spec: specs/VLA-TASK-SPEC.md 1.1-1.4).

Layering
--------
* physics:    MuJoCo 0.005 s timestep (sim/scene_vla.xml)
* low level:  SB3 PPO policy from assets/lowlevel, 0.01 s decision rate (100 Hz),
              fed with the VecNormalize obs statistics (normalize_obs) as required by the spec.
              The policy consumes (v_cmd, w_cmd) and outputs wheel acceleration commands.
* VLA:        0.1 s decision rate (10 Hz), action = (v, w) physical units, clipped to the
              spec action space v in [-0.35, 0.35] m/s, w in [-1.0, 1.0] rad/s.
* episode:    fixed 50 VLA steps = 5.0 s, early failure on fall (pitch > 0.4 or z < 0.012).

Scene (per episode, deterministic from `seed`)
----------------------------------------------
* 8 m x 8 m checker floor, 4 low walls, sky gradient, head camera on the chassis
  (128 x 128, fovy 70 deg, ~0.18 m above ground, 15 deg down).
* 3 buckets (r=0.07 m, h=0.22 m) red / blue / green.  Positions are polar around the
  robot start pose: radius 0.65-1.4 m, azimuth +-30 deg, pairwise distance >= 0.45 m,
  so every bucket is inside the +-35 deg camera field of view at t=0.
* two scene layouts, chosen by seed:
    - generic  (seed % 6 != 5): fully random positions, random colour permutation;
    - obstacle (seed % 6 == 5): used for instruction 11 -- the blue bucket is sampled
      uniformly, the red bucket is placed 0.7-1.0 m from the robot at a lateral offset
      of 0.12-0.25 m from the robot->blue line (minimum offset raised from 0 to keep the
      blue bucket visible behind the red one; see sim/README.md), green is random.
  The dataset generator pairs instruction 0 ("stand") with instruction 11 on those
  obstacle seeds, so every scene can host every instruction.
* per-episode randomisation: bucket positions + colour assignment + start pose
  (x,y +-0.1 m, yaw +-0.2 rad) + light strength (0.7-1.3) + floor brightness (0.85-1.15)
  + camera noise (sigma 2.0 / 255).
* same seed -> identical scene (bucket slots/colours/lighting/start pose).

Oracle / expert support
-----------------------
`info["dist"]` and `info["heading_err_by_color"]` are keyed by colour id
(0 = red, 1 = blue, 2 = green) and are computed from privileged ground truth.
`info["cum_dyaw"]` is the frame-by-frame unwrapped heading change (never the wrapped
first/last difference -- that is wrong for turns beyond +-pi, see spec 1.3).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# MuJoCo must be told to use EGL before it is imported: the machine has no working osmesa
import os

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
_LOWLEVEL = ROOT / "assets" / "lowlevel"
if str(_LOWLEVEL) not in sys.path:
    sys.path.insert(0, str(_LOWLEVEL))

import mujoco  # noqa: E402
import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import env_cmd3d  # noqa: E402

SCENE_XML = ROOT / "sim" / "scene_vla.xml"
env_cmd3d.XML = str(SCENE_XML)

IMAGE_SIZE = 128
GROUND_CLEARANCE = 0.11          # bucket body z so that the cylinder bottom sits on the floor
BUCKET_RADIUS = 0.07
BUCKET_HEIGHT = 0.22
ACTION_LOW = np.array([-0.35, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([0.35, 1.0], dtype=np.float32)
ACTION_LIMIT = 0.35
OMEGA_LIMIT = 1.0

COLORS = ("red", "blue", "green")
COLOR_ID = {"red": 0, "blue": 1, "green": 2}

INSTRUCTIONS = [
    "保持站立",
    "向前走",
    "向后退",
    "向左转",
    "向右转",
    "原地左旋",
    "原地右旋",
    "面向红桶",
    "靠近红桶",
    "靠近蓝桶",
    "靠近绿桶",
    "绕开红桶前往蓝桶",
]


def wrap_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class _ObsStub(gym.Env):
    """Minimal gym env only used to instantiate VecNormalize for obs normalisation."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-10.0, 10.0, (9,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(9, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(9, dtype=np.float32), 0.0, False, False, {}


def success_func(instr_id, info, history):
    """Success criterion of specs/VLA-TASK-SPEC.md 1.3, evaluated with the info at the
    moment the episode ends (t = 50 or fall).  `history` holds all per-step info dicts,
    history[-1] is the current step (the same dict as `info` in normal use).

    The function is deterministic and side-effect free; it is also called every step to
    populate info["success"] ("would succeed if the episode ended now").
    """
    if info.get("fall", False):
        return False

    start = np.asarray(info["start_pos"], dtype=np.float64)
    pos = np.asarray(info["pos"], dtype=np.float64)
    start_yaw = float(info["start_yaw"])
    delta = pos - start
    forward = float(delta[0] * math.cos(start_yaw) + delta[1] * math.sin(start_yaw))
    distance = float(np.hypot(delta[0], delta[1]))
    dyaw = float(info["cum_dyaw"])
    dist = info["dist"]
    heading = info["heading_err_by_color"]

    if instr_id == 0:
        return bool(distance < 0.10 and abs(dyaw) < 0.20)
    if instr_id == 1:
        return bool(0.50 <= forward <= 1.90)
    if instr_id == 2:
        return bool(0.35 <= -forward <= 1.50)
    if instr_id == 3:
        return bool(1.0 <= dyaw <= 3.5 and forward > 0.15)
    if instr_id == 4:
        return bool(-3.5 <= dyaw <= -1.0 and forward > 0.15)
    if instr_id == 5:
        return bool(1.5 <= dyaw <= 5.0 and distance < 0.40)
    if instr_id == 6:
        return bool(-5.0 <= dyaw <= -1.5 and distance < 0.40)
    if instr_id == 7:
        return bool(abs(heading[0]) < 0.30)
    if instr_id == 8:
        return bool(dist[0] < 0.45)
    if instr_id == 9:
        return bool(dist[1] < 0.45)
    if instr_id == 10:
        return bool(dist[2] < 0.45)
    if instr_id == 11:
        if dist[1] >= 0.50:
            return False
        min_red = min(float(h["dist"][0]) for h in (history if history else [info]))
        return bool(min_red > 0.40)
    raise ValueError("unknown instruction id: %r" % (instr_id,))


def _pairwise_min(points):
    """Minimum pairwise distance of a list of 2D points."""
    n = len(points)
    best = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, float(np.hypot(*(np.asarray(points[i]) - np.asarray(points[j])))))
    return best


class VLAEnv:
    """VLA-facing simulation environment, see module docstring.

    Parameters
    ----------
    instr_id, seed : accepted by reset() (spec 1.4), both optional.
    max_steps      : VLA steps per episode (spec: 50 -> 5.0 s).
    render         : create an EGL renderer (needed unless only physics state is used).
    load_policy    : load the SB3 low-level policy + VecNormalize (set False for scene
                     inspection only).
    model_path, vec_path : override the spec assets (must be a 9-dim Cmd3D model).
    angle_max_deg, radius_min, radius_max, min_sep, obstacle_mod :
                     scene layout knobs, defaults match the updated spec 1.2.
    init_pitch_max : initial pitch randomisation of the low-level plant (rad).
    noise_sigma    : per-frame RGB camera noise std in 0..255 units.
    collide        : keep the bucket collision geoms enabled (default True).  Set False to
                     run the ablation where the robot can drive through the buckets.
    """

    INSTRUCTIONS = list(INSTRUCTIONS)
    COLORS = COLORS
    COLOR_ID = dict(COLOR_ID)
    obs_dim_image = (IMAGE_SIZE, IMAGE_SIZE, 3)
    action_dim = 2
    max_steps = 50

    def __init__(
        self,
        max_steps=50,
        render=True,
        load_policy=True,
        model_path=None,
        vec_path=None,
        device="cpu",
        angle_max_deg=30.0,
        radius_min=0.65,
        radius_max=1.4,
        min_sep=0.45,
        init_pitch_max=0.03,
        noise_sigma=2.0,
        collide=True,
        obstacle_mod=6,
        obstacle_phase=5,
        obstacle_lateral_min=0.12,
        obstacle_lateral_max=0.25,
        obstacle_radius_min=0.70,
        obstacle_radius_max=1.00,
        image_size=IMAGE_SIZE,
    ):
        self.max_steps = int(max_steps)
        self.angle_max_deg = float(angle_max_deg)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.min_sep = float(min_sep)
        self.noise_sigma = float(noise_sigma)
        self.obstacle_lateral_min = float(obstacle_lateral_min)
        self.obstacle_lateral_max = float(obstacle_lateral_max)
        self.obstacle_radius_min = float(obstacle_radius_min)
        self.obstacle_radius_max = float(obstacle_radius_max)
        self.image_size = int(image_size)
        self.device = device
        self.collide = bool(collide)
        self.obstacle_mod = max(int(obstacle_mod), 1)
        self.obstacle_phase = int(obstacle_phase) % self.obstacle_mod

        # ---- plant: the low-level policy owns model + data -------------------------------
        self.low = env_cmd3d.Cmd3D(
            seed=0,
            max_steps=10 ** 9,
            mode="stand",
            vmax=ACTION_LIMIT,
            wmax=OMEGA_LIMIT,
            zero_frac=0.0,
            randomize=False,
            init_pitch_max=init_pitch_max,
        )
        self.model = self.low.m
        self.data = self.low.d
        self.init_pitch_max = float(init_pitch_max)

        # ---- ids -----------------------------------------------------------------------
        self.bucket_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bucket%d" % i) for i in range(3)
        ]
        self.bucket_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bucket%d_geom" % i) for i in range(3)
        ]
        self._mat_ids = {
            color: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "mat_" + color)
            for color in COLORS
        }
        self._floor_mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "floor_mat")
        self._base_floor_rgba = self.model.mat_rgba[self._floor_mat_id].copy()
        self._base_headlight_diffuse = self.model.vis.headlight.diffuse.copy()
        self._base_headlight_ambient = self.model.vis.headlight.ambient.copy()
        self._base_light_diffuse = self.model.light_diffuse.copy()
        if min(self.bucket_body_ids) < 0 or min(self.bucket_geom_ids) < 0:
            raise RuntimeError("scene_vla.xml is missing bucket bodies/geoms")
        if not self.collide:
            # ablation only: bucket geoms become visual-only (dataset default is collide=True)
            for gid in self.bucket_geom_ids:
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0

        # ---- renderer ------------------------------------------------------------------
        self.renderer = mujoco.Renderer(self.model, self.image_size, self.image_size) if render else None
        self._segmentation = False

        # ---- low-level policy + VecNormalize obs statistics ----------------------------
        self.policy = None
        self._obs_mean = None
        self._obs_scale = None
        self._policy_obs = None
        if load_policy:
            mp = Path(model_path) if model_path else (_LOWLEVEL / "balance_model.zip")
            vp = Path(vec_path) if vec_path else (_LOWLEVEL / "vecnormalize.pkl")
            self.policy = PPO.load(str(mp), device=device)
            stub = DummyVecEnv([lambda: _ObsStub()])
            self.norm = VecNormalize.load(str(vp), stub)
            self.norm.training = False
            self.norm.norm_reward = False
            self._obs_mean = np.asarray(self.norm.obs_rms.mean, dtype=np.float32).copy()
            self._obs_scale = (1.0 / np.sqrt(np.asarray(self.norm.obs_rms.var, dtype=np.float32) + 1e-8)).astype(
                np.float32
            )
            self._obs_clip = float(self.norm.clip_obs)
            self._policy_obs = __import__("torch").zeros((1, 9), dtype=__import__("torch").float32)

        # ---- episode state -------------------------------------------------------------
        self.t = 0
        self.done = True
        self.instr_id = 0
        self.scene = None
        self.image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.history = []
        self.cum_dyaw = 0.0
        self._prev_yaw = 0.0
        self._episode_seed = 0
        self._scene_rng = np.random.default_rng(0)
        self._noise_rng = None
        self._fall = False
        self._last_render_step = -1
        self.last_step_ms = 0.0

    # ------------------------------------------------------------------ scene generation
    @staticmethod
    def scene_mode_for_seed(seed, obstacle_mod=6, obstacle_phase=5):
        """Layout mode derived from the scene seed (deterministic).

        Legacy datasets (dataset_v1 / dataset_test) use obstacle_mod=6, obstacle_phase=5,
        i.e. exactly ``seed % 6 == 5``; the object-only dataset uses obstacle_mod=3,
        obstacle_phase=0 so that every third seed is an obstacle scene.  Both rules are
        seed-deterministic, which keeps ``reset(seed)`` reproducible.
        """
        mod = max(int(obstacle_mod), 1)
        phase = int(obstacle_phase) % mod
        return "obstacle" if int(seed) % mod == phase else "generic"

    def build_scene(self, seed):
        """Create the (deterministic) scene description for `seed` without touching physics."""
        rng = np.random.default_rng(int(seed))
        start_x = float(rng.uniform(-0.1, 0.1))
        start_y = float(rng.uniform(-0.1, 0.1))
        start_yaw = float(rng.uniform(-0.2, 0.2))
        light_scale = float(rng.uniform(0.7, 1.3))
        floor_scale = float(rng.uniform(0.85, 1.15))
        mode = self.scene_mode_for_seed(seed, self.obstacle_mod, self.obstacle_phase)

        if mode == "obstacle":
            rel, colors = self._obstacle_layout(rng)
        else:
            rel, colors = self._generic_layout(rng)

        cos_y, sin_y = math.cos(start_yaw), math.sin(start_yaw)
        world = []
        for px, py in rel:
            world.append([start_x + cos_y * px - sin_y * py, start_y + sin_y * px + cos_y * py])

        return {
            "scene_seed": int(seed),
            "mode": mode,
            "start_xy": [start_x, start_y],
            "start_yaw": start_yaw,
            "light_scale": light_scale,
            "floor_scale": floor_scale,
            "positions": world,
            "colors": [int(c) for c in colors],
            "bucket_radius": BUCKET_RADIUS,
            "bucket_height": BUCKET_HEIGHT,
        }

    def _generic_layout(self, rng):
        """Three random polar slots + random colour permutation."""
        angle = math.radians(self.angle_max_deg)
        best = None
        for _ in range(400):
            pts = []
            for _slot in range(3):
                a = float(rng.uniform(-angle, angle))
                r = float(rng.uniform(self.radius_min, self.radius_max))
                pts.append((r * math.cos(a), r * math.sin(a)))
            sep = _pairwise_min(pts)
            if best is None or sep > best[1]:
                best = (pts, sep)
            if sep >= self.min_sep:
                return pts, list(rng.permutation(3))
        # deterministic fallback (checked to satisfy the separation constraint)
        pts = [(-0.5, -0.78), (0.0, 1.40), (0.40, -0.66)]
        return pts, list(rng.permutation(3))

    def _obstacle_layout(self, rng):
        """Instruction-11 layout: red bucket near the robot -> blue bucket line."""
        angle = math.radians(self.angle_max_deg)
        best = None
        for _ in range(400):
            a_b = float(rng.uniform(-angle, angle))
            r_b = float(rng.uniform(self.radius_min, self.radius_max))
            blue = (r_b * math.cos(a_b), r_b * math.sin(a_b))
            lateral = float(rng.uniform(-self.obstacle_lateral_max, self.obstacle_lateral_max))
            if abs(lateral) < self.obstacle_lateral_min:
                lateral = math.copysign(self.obstacle_lateral_min, lateral if lateral != 0.0 else rng.choice([-1.0, 1.0]))
            d_red = float(rng.uniform(self.obstacle_radius_min, self.obstacle_radius_max))
            along = math.sqrt(max(d_red * d_red - lateral * lateral, 0.01))
            red = (along * math.cos(a_b) - lateral * math.sin(a_b),
                   along * math.sin(a_b) + lateral * math.cos(a_b))
            # keep the red bucket inside the camera view as well
            if abs(math.atan2(red[1], red[0])) > angle + 0.02:
                red = (along * math.cos(a_b) + lateral * math.sin(a_b),
                       along * math.sin(a_b) - lateral * math.cos(a_b))
            red_blue = float(np.hypot(red[0] - blue[0], red[1] - blue[1]))
            if red_blue < self.min_sep:
                continue
            green = None
            for _try in range(200):
                a_g = float(rng.uniform(-angle, angle))
                r_g = float(rng.uniform(self.radius_min, self.radius_max))
                cand = (r_g * math.cos(a_g), r_g * math.sin(a_g))
                if min(float(np.hypot(cand[0] - blue[0], cand[1] - blue[1])),
                       float(np.hypot(cand[0] - red[0], cand[1] - red[1]))) >= self.min_sep:
                    green = cand
                    break
            if green is None:
                continue
            pts = [blue, red, green]
            score = min(_pairwise_min(pts), red_blue)
            if best is None or score > best[1]:
                best = (pts, score)
            if score >= self.min_sep:
                return pts, [COLOR_ID["blue"], COLOR_ID["red"], COLOR_ID["green"]]
        if best is not None:
            return best[0], [COLOR_ID["blue"], COLOR_ID["red"], COLOR_ID["green"]]
        pts = [(-0.30, 1.10), (0.10, 0.85), (0.55, -0.60)]
        return pts, [COLOR_ID["blue"], COLOR_ID["red"], COLOR_ID["green"]]

    # ------------------------------------------------------------------ scene application
    def _apply_scene(self, scene):
        for slot in range(3):
            x, y = scene["positions"][slot]
            self.model.body_pos[self.bucket_body_ids[slot]] = [x, y, GROUND_CLEARANCE]
            self.model.geom_matid[self.bucket_geom_ids[slot]] = self._mat_ids[COLORS[scene["colors"][slot]]]

        scale = float(scene["floor_scale"])
        rgba = self._base_floor_rgba.copy()
        rgba[:3] = np.clip(rgba[:3] * scale, 0.0, 1.0)
        self.model.mat_rgba[self._floor_mat_id] = rgba

        light = float(scene["light_scale"])
        self.model.vis.headlight.diffuse[:] = np.clip(self._base_headlight_diffuse * light, 0.0, 1.0)
        self.model.vis.headlight.ambient[:] = np.clip(self._base_headlight_ambient * light, 0.0, 1.0)
        self.model.light_diffuse[:] = np.clip(self._base_light_diffuse * light, 0.0, 1.0)

    def _apply_start_pose(self, scene):
        x, y = scene["start_xy"]
        yaw = scene["start_yaw"]
        q = self.data.qpos
        q[0] = x
        q[1] = y
        q_yaw = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        q_pitch = q[3:7].copy()
        mujoco.mju_mulQuat(q[3:7], q_yaw, q_pitch)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ observation
    def _low_policy_action(self, obs9):
        """One 100 Hz low-level action from the raw 9-dim Cmd3D observation."""
        obs = (np.asarray(obs9, dtype=np.float32) - self._obs_mean) * self._obs_scale
        np.clip(obs, -self._obs_clip, self._obs_clip, out=obs)
        torch = __import__("torch")
        self._policy_obs[0] = torch.from_numpy(obs)
        with torch.no_grad():
            action = self.policy.policy(self._policy_obs, deterministic=True)[0]
        return action.numpy()[0]

    def _pose(self):
        pitch, yaw, yawrate, vx, vy = self.low._state()
        return pitch, yaw, yawrate, vx, vy

    def _bucket_distances(self):
        pos = np.asarray(self.data.qpos[0:2], dtype=np.float64)
        yaw = self._yaw_now
        dists = {}
        headings = {}
        for slot in range(3):
            color = self.scene["colors"][slot]
            bx, by = self.scene["positions"][slot]
            dx, dy = bx - pos[0], by - pos[1]
            dists[color] = float(math.hypot(dx, dy))
            headings[color] = wrap_angle(math.atan2(dy, dx) - yaw)
        return dists, headings

    def _make_info(self, t):
        pos = np.asarray(self.data.qpos[0:2], dtype=np.float64)
        pitch, yaw, yawrate, vx, vy = self._pose()
        self._yaw_now = yaw
        dists, headings = self._bucket_distances()
        info = {
            "dist": dists,
            "heading_err_by_color": headings,
            "success": False,
            "fall": bool(self._fall),
            "pos": (float(pos[0]), float(pos[1])),
            "yaw": float(yaw),
            "vx": float(vx),
            "vy": float(vy),
            "yawrate": float(yawrate),
            "pitch": float(pitch),
            "z": float(self.data.qpos[2]),
            "cum_dyaw": float(self.cum_dyaw),
            "start_pos": tuple(self.scene["start_xy"]),
            "start_yaw": float(self.scene["start_yaw"]),
            "instr_id": int(self.instr_id),
            "t": int(t),
            "scene_seed": int(self._episode_seed),
            "scene_mode": self.scene["mode"],
            "bucket_positions": [list(p) for p in self.scene["positions"]],
            "bucket_colors": [COLORS[c] for c in self.scene["colors"]],
        }
        info["success"] = success_func(self.instr_id, info, self.history + [info])
        return info

    # ------------------------------------------------------------------ rendering
    def render_rgb(self):
        """Render the current 128x128 RGB frame from the head camera (uint8)."""
        if self.renderer is None:
            raise RuntimeError("VLAEnv was created with render=False")
        self.renderer.update_scene(self.data, camera="vla_cam")
        return self.renderer.render()

    def capture(self, add_noise=True):
        """Render one frame (optionally with the per-episode camera noise)."""
        img = self.render_rgb()
        if add_noise and self.noise_sigma > 0:
            noise = self._noise_rng.normal(0.0, self.noise_sigma, size=img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        return img

    def segmentation(self):
        """Render the current frame as a segmentation mask (object id, object type)."""
        if self.renderer is None:
            raise RuntimeError("VLAEnv was created with render=False")
        if not self._segmentation:
            self.renderer.enable_segmentation_rendering()
            self._segmentation = True
        self.renderer.update_scene(self.data, camera="vla_cam")
        return self.renderer.render().copy()

    def segmentation_off(self):
        if self.renderer is not None and self._segmentation:
            self.renderer.disable_segmentation_rendering()
            self._segmentation = False

    def bucket_pixel_counts(self):
        """Pixels per bucket colour in the current camera view (via segmentation)."""
        seg = self.segmentation()
        counts = {}
        for slot in range(3):
            gid = self.bucket_geom_ids[slot]
            mask = (seg[:, :, 0] == gid) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            counts[COLORS[self.scene["colors"][slot]]] = int(mask.sum())
        self.segmentation_off()
        return counts

    # ------------------------------------------------------------------ env API
    def reset(self, instr_id=None, seed=None):
        """Start a new episode; returns the first 128x128x3 uint8 image.

        The returned image (and every image returned by step()) is the camera image **with**
        the spec 1.2 noise (sigma=2, deterministic from the seed); use render_rgb() for the
        clean frame.  Demo / eval consume the same noisy image as the dataset.
        """
        if seed is None:
            seed = int(self._episode_seed) + 1
        seed = int(seed)
        self._episode_seed = seed
        if instr_id is None:
            instr_id = int(np.random.default_rng(seed).integers(0, len(INSTRUCTIONS)))
        instr_id = int(instr_id)
        if not 0 <= instr_id < len(INSTRUCTIONS):
            raise ValueError("instr_id must be in [0, 11], got %r" % (instr_id,))
        self.instr_id = instr_id

        # deterministic physics init (initial pitch) + scene
        self.low.rng = np.random.default_rng(seed)
        self.low.reset()
        self.scene = self.build_scene(seed)
        self._apply_scene(self.scene)
        self._apply_start_pose(self.scene)

        self._noise_rng = np.random.default_rng([seed, 0x5EED])
        self.t = 0
        self.done = False
        self._fall = False
        self.cum_dyaw = 0.0
        self._prev_yaw = float(self._pose()[1])
        self._yaw_now = self._prev_yaw
        self.history = []
        info = self._make_info(0)
        self.history.append(info)
        self.image = self.capture()
        return self.image

    def step(self, action_vw):
        """Apply one 10 Hz VLA action (v, w); returns (image, done, info)."""
        if self.scene is None:
            raise RuntimeError("call reset() before step()")
        action = np.clip(np.asarray(action_vw, dtype=np.float32).reshape(2), ACTION_LOW, ACTION_HIGH)
        v_cmd = float(action[0])
        w_cmd = float(action[1])
        fall = False
        for _ in range(10):
            self.low.v_cmd = v_cmd
            self.low.w_cmd = w_cmd
            low_action = self._low_policy_action(self.low._obs())
            _obs, _rew, _done, low_info = self.low.step(low_action)
            if abs(float(low_info["pitch"])) > 0.4 or float(self.data.qpos[2]) < 0.012:
                fall = True
                break
        self._fall = fall

        yaw = float(self._pose()[1])
        self.cum_dyaw += wrap_angle(yaw - self._prev_yaw)
        self._prev_yaw = yaw
        self.t += 1
        info = self._make_info(self.t)
        self.history.append(info)
        self.done = bool(fall or self.t >= self.max_steps)
        self.image = self.capture()
        info["action"] = (v_cmd, w_cmd)
        return self.image, self.done, info

    # ------------------------------------------------------------------ helpers
    def close(self):
        self.segmentation_off()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def scene_info(self):
        """JSON-serialisable description of the current episode scene."""
        if self.scene is None:
            return None
        return dict(self.scene)
