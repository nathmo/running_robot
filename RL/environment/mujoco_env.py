"""
MuJoCo-based legged robot environment with Gymnasium interface
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path as PathLib
import os

from .terrain import TerrainGenerator
from .paths import PathTracker, create_random_path, CircularPath, StraightPath


class LeggedRobotEnv(gym.Env):
    """
    Gymnasium environment for training RL policies on legged robots
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config,
        urdf_path=None,
        render_mode=None,
        terrain_seed=None,
        randomize_path=True,
    ):
        """
        Args:
            config: Configuration dict (from config.py)
            urdf_path: Override URDF path if needed
            render_mode: "human" or "rgb_array"
            terrain_seed: Specific seed for terrain (overrides config)
            randomize_path: Whether to randomize path each episode
        """
        self.config = config
        self.render_mode = render_mode
        self.randomize_path = randomize_path
        self.terrain_seed = terrain_seed or config["TERRAIN"]["seed"]

        # Load URDF and create MuJoCo model
        self.urdf_path = urdf_path or config["ROBOT"]["urdf_path"]
        self._load_model()

        # Terrain generator
        self.terrain_gen = TerrainGenerator(
            seed=self.terrain_seed,
            grid_size=config["TERRAIN"]["grid_size"],
            grid_spacing=config["TERRAIN"]["grid_spacing"],
        )
        self.heightfield = self._generate_terrain()

        # Path tracker
        self.current_path = None
        self._reset_path()

        # Create Gymnasium spaces
        self._setup_spaces()

        # State tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_base_pos = np.array([0.0, 0.0, 0.0])
        self.prev_action = np.zeros(self.action_dim)

        # Feet-air-time tracking (one entry per foot geom).
        # `foot_airtime[i]` is seconds since foot i last made contact.
        # `foot_prev_contact[i]` is whether foot i was in contact at the previous control step.
        n_feet = len(self.foot_geom_ids)
        self.foot_airtime = np.zeros(n_feet, dtype=np.float32)
        self.foot_prev_contact = np.zeros(n_feet, dtype=bool)

        # Rendering
        self.viewer = None

    def _load_model(self):
        """Load robot URDF model"""
        # Resolve path relative to this file or as absolute
        urdf_path = PathLib(self.urdf_path)
        if not urdf_path.is_absolute():
            # Try relative to this file first
            base_dir = PathLib(__file__).parent.parent
            urdf_path = base_dir / self.urdf_path

        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF/MJCF not found: {urdf_path}")

        # Load MuJoCo model (handles both URDF and MJCF)
        self.model = mujoco.MjModel.from_xml_path(str(urdf_path))
        self.data = mujoco.MjData(self.model)

        # Extract actuator info (MuJoCo 3.x API)
        self.action_dim = self.model.nu  # Number of actuators
        self.num_dofs = self.model.nv
        self.num_bodies = self.model.nbody

        print(
            f"Loaded model: {self.action_dim} actuators, {self.num_dofs} DOFs, {self.num_bodies} bodies"
        )

        # Store reference to base link
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.foot_body_ids = []

        # Find foot bodies by name (MuJoCo 3.x: try name-based lookup)
        self.foot_body_ids = []
        for body_name in ["left_foot", "right_foot"]:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                self.foot_body_ids.append(body_id)

        if not self.foot_body_ids:
            # Fallback: use last few bodies as feet
            self.foot_body_ids = list(range(max(0, self.model.nbody - 4), self.model.nbody))

        # Foot *geom* ids (needed for contact detection; geoms and bodies share names here).
        self.foot_geom_ids = []
        for geom_name in ["left_foot", "right_foot"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if gid >= 0:
                self.foot_geom_ids.append(gid)

        # Ground geom id — used by fall detection to check whether any non-foot
        # robot part is touching the floor (torso/thigh/calf contact = fall on
        # the real robot → damage).
        self.ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground"
        )

        print(
            f"Base body ID: {self.base_body_id}, "
            f"Foot body IDs: {self.foot_body_ids}, "
            f"Foot geom IDs: {self.foot_geom_ids}, "
            f"Ground geom ID: {self.ground_geom_id}"
        )

        # Pre-compute the qvel index for each actuator's joint. Used by the motor
        # saturation model in step() to enforce the velocity limit per-joint.
        self.actuator_dof_indices = []
        for act_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[act_id, 0])
            if joint_id >= 0:
                self.actuator_dof_indices.append(int(self.model.jnt_dofadr[joint_id]))
            else:
                self.actuator_dof_indices.append(-1)

    def _setup_spaces(self):
        """Define observation and action spaces"""
        # Action space: motor commands
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # Observation space: compute from actual model structure.
        # qpos includes all joint positions (freejoint is 7D: 3 pos + 4 quat);
        # we drop the absolute (x, y) world position because running forward is
        # translationally invariant — those two dims are pure noise the policy
        # has to learn to ignore. Height (z) is kept because it matters.
        qpos_dim = self.model.nq - 2  # drop x, y
        qvel_dim = self.model.nv      # keep all velocities

        obs_dim = (
            qpos_dim
            + qvel_dim
            + 4  # base quaternion
            + 3  # base linear velocity
            + 3  # base angular velocity
            + self.action_dim  # previous action
        )

        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype=np.float32
        )

        print(f"Observation space: {obs_dim} dims (qpos={qpos_dim}, qvel={qvel_dim}, action={self.action_dim})")

    def _generate_terrain(self):
        """Generate heightfield terrain"""
        terrain_config = self.config["TERRAIN"]

        if terrain_config["type"] == "perlin":
            return self.terrain_gen.generate_perlin(
                scale=terrain_config["perlin_scale"],
                octaves=terrain_config["perlin_octaves"],
                persistence=terrain_config["perlin_persistence"],
                lacunarity=terrain_config["perlin_lacunarity"],
                height_scale=terrain_config["height_scale"],
                height_offset=terrain_config["height_offset"],
            )
        elif terrain_config["type"] == "flat":
            return self.terrain_gen.generate_flat(height=terrain_config["height_offset"])
        elif terrain_config["type"] == "stairs":
            return self.terrain_gen.generate_stairs()
        else:
            raise ValueError(f"Unknown terrain type: {terrain_config['type']}")

    def _reset_path(self):
        """Reset to a new path.

        When `PATHS.straight_line` is true, the path is always an infinite straight
        line along +x (straight-running task). Otherwise falls back to circles of
        varying radii.
        """
        path_config = self.config["PATHS"]

        if path_config.get("straight_line", False):
            self.current_path = StraightPath(direction=(1.0, 0.0))
        elif self.randomize_path:
            radius = np.random.choice(path_config["radii"])
            self.current_path = CircularPath(radius=radius)
        else:
            self.current_path = CircularPath(radius=path_config["radii"][0])

        self.path_tracker = PathTracker(self.current_path)

    def _get_observation(self):
        """Construct observation vector.

        We drop qpos[0:2] (absolute world x, y) — see _setup_spaces for the
        rationale. Everything else stays: z, quat, joint angles, all velocities,
        base kinematics, and previous action.
        """
        # Joint positions and velocities. Slice off absolute x,y.
        qpos = self.data.qpos[2:].copy()
        qvel = self.data.qvel.copy()

        # Base pose and velocity
        base_quat = self.data.body(self.base_body_id).xquat.copy()
        base_vel = self.data.body(self.base_body_id).cvel[:3].copy()
        base_ang_vel = self.data.body(self.base_body_id).cvel[3:].copy()

        # Construct obs
        obs = np.concatenate(
            [
                qpos,
                qvel,
                base_quat,
                base_vel,
                base_ang_vel,
                self.prev_action,
            ]
        ).astype(np.float32)

        return obs

    def _get_foot_contacts(self):
        """Return a boolean array: True if foot i is currently in contact with anything.

        We scan active contacts at the end of the control step. A foot-geom appearing
        as either side of a contact pair counts as "in contact".
        """
        n_feet = len(self.foot_geom_ids)
        if n_feet == 0:
            return np.zeros(0, dtype=bool)

        contacts = np.zeros(n_feet, dtype=bool)
        foot_set = set(self.foot_geom_ids)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in foot_set:
                contacts[self.foot_geom_ids.index(g1)] = True
            if g2 in foot_set:
                contacts[self.foot_geom_ids.index(g2)] = True
        return contacts

    def _update_feet_air_time(self, v_along):
        """Update per-foot airtime counters and return the ANYmal/Rudin airtime reward.

        At the moment a foot transitions air -> contact, credit
            (airtime_at_landing - threshold) * weight
        Gated off at very low forward speeds so the policy can't farm the term by
        marching in place without travelling.
        """
        r = self.config["REWARD"]
        n_feet = len(self.foot_geom_ids)
        if n_feet == 0:
            return 0.0

        contacts_now = self._get_foot_contacts()
        # Landing event = was airborne last step, is in contact now.
        first_contact = contacts_now & (~self.foot_prev_contact)

        # Compute reward from each landing. We use the airtime accumulated up to this
        # step (before we zero it out below).
        threshold = r["feet_air_time_threshold"]
        gate = 1.0 if v_along > r["feet_air_time_min_speed"] else 0.0
        landing_reward = float(
            np.sum((self.foot_airtime[first_contact] - threshold))
            * r["feet_air_time_weight"]
            * gate
        )

        # Advance the bookkeeping.
        control_dt = self.config["ROBOT"]["control_dt"]
        self.foot_airtime[~contacts_now] += control_dt
        self.foot_airtime[contacts_now] = 0.0
        self.foot_prev_contact = contacts_now

        return landing_reward

    def _get_reward(self):
        """Compute reward for this step.

        Terms (all from REWARD config):
          +forward    min(v_along_path, target_speed) * forward_speed_weight  [m/s-scaled]
          -upright    (1 - body_z . world_z) * upright_weight                 [0 upright, 2 flipped]
          -smoothness ||a_t - a_{t-1}|| * action_smoothness_weight
          -deviation  distance_from_path * track_deviation_weight             [m]
          +alive      alive_bonus
        `fall_penalty` is applied once by step() when termination is caused by a fall.
        """
        r = self.config["REWARD"]

        base_pos = self.data.body(self.base_body_id).xpos.copy()

        # Forward speed in m/s, measured along the path tangent at the prev position.
        # For StraightPath this is just dx/dt along the path direction.
        dt = self.config["ROBOT"]["control_dt"]
        step_vec = base_pos[:2] - self.prev_base_pos[:2]
        _, _, t_prev = self.current_path.get_closest_point(
            float(self.prev_base_pos[0]), float(self.prev_base_pos[1])
        )
        heading = self.current_path.get_heading(t_prev)
        tangent = np.array([np.cos(heading), np.sin(heading)])
        v_along = float(np.dot(step_vec, tangent)) / max(dt, 1e-9)  # m/s
        # Cache so step() can surface the true speed in info (used by eval metrics).
        self._last_v_along = v_along

        forward_reward = min(v_along, r["forward_target_speed"]) * r["forward_speed_weight"]

        # Upright penalty via body-z projection onto world-z.
        # xmat is row-major 3x3; last column = body z-axis in world frame.
        xmat = self.data.body(self.base_body_id).xmat.reshape(3, 3)
        body_z_world_z = float(xmat[2, 2])
        upright_penalty = (1.0 - body_z_world_z) * r["upright_weight"]

        # Action smoothness
        action_diff = float(np.linalg.norm(self._last_action - self.prev_action))
        smoothness_penalty = action_diff * r["action_smoothness_weight"]

        # Path deviation (m). For StraightPath this is perpendicular distance to the line.
        deviation = float(self.path_tracker.get_deviation(base_pos[0], base_pos[1]))
        deviation_penalty = deviation * r["track_deviation_weight"]

        # Feet-air-time reward (ANYmal/Rudin). Fires only on landing events; zero most steps.
        air_time_reward = self._update_feet_air_time(v_along)

        total = (
            forward_reward
            - upright_penalty
            - smoothness_penalty
            - deviation_penalty
            + air_time_reward
            + r["alive_bonus"]
        )
        return float(total)

    def _non_foot_ground_contact(self):
        """Return True if any non-foot robot geom is touching the ground.

        The real robot is damaged if the torso, thighs, or calves hit the floor
        — only the feet are supposed to make ground contact. We catch this
        directly via the MuJoCo contact list so the policy cannot exploit a
        "flop on back and wiggle" gait that previously stayed within the
        base_z > 0.1 termination threshold.
        """
        if self.ground_geom_id < 0:
            return False
        foot_set = set(self.foot_geom_ids)
        ground = self.ground_geom_id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == ground and g2 not in foot_set:
                return True
            if g2 == ground and g1 not in foot_set:
                return True
        return False

    def _check_termination(self):
        """Return (terminated, reason) where reason is 'fall', 'off_path', 'timeout' or None."""
        base_pos = self.data.body(self.base_body_id).xpos

        # Any non-foot robot geom contacting the ground = fall (torso scrape,
        # knee-down, etc.). Strong termination signal against self-damaging gaits.
        if self._non_foot_ground_contact():
            return True, "fall"

        if base_pos[2] < 0.1:
            return True, "fall"

        deviation = self.path_tracker.get_deviation(base_pos[0], base_pos[1])
        if deviation > 5.0:
            return True, "off_path"

        if self.step_count >= self.config["RL"]["max_episode_steps"]:
            return True, "timeout"

        return False, None

    def _is_done(self):
        """Legacy: kept for any external callers. Prefer _check_termination()."""
        terminated, _ = self._check_termination()
        return terminated

    def reset(self, seed=None, options=None):
        """Reset environment.

        Spawn the robot upright with feet just clearing the ground, then settle for
        a few sim steps so contact transients die out before the policy sees its
        first obs. The model's default base z (0.5 for simple_biped) leaves the
        feet penetrating the ground by ~0.11m; constraint forces would then launch
        the robot into a near-random orientation, making early learning much
        harder than necessary.
        """
        super().reset(seed=seed)

        # Reset path occasionally
        if self.randomize_path and np.random.rand() < 0.3:
            self._reset_path()

        # Reset to default pose
        self.data.qpos = self.model.qpos0.copy()
        self.data.qvel = np.zeros(self.num_dofs)

        # Place base upright at a height where the feet just clear the ground.
        # qpos layout for the freejoint: [x, y, z, qw, qx, qy, qz], then joint angles.
        spawn_height = self.config["ROBOT"].get("spawn_height", 0.62)
        self.data.qpos[:3] = [0.0, 0.0, spawn_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Small noise on joint angles only — never on base pos or quaternion,
        # which used to produce non-unit quats and tumbling spawns.
        n_joints = self.model.nq - 7
        if n_joints > 0:
            self.data.qpos[7:7 + n_joints] += np.random.normal(0, 0.01, n_joints)

        # Settle physics so ground contact converges before the first obs.
        # 50 sim steps at sim_dt=0.001 ≈ 50 ms of settling.
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        # Start the episode from rest so settling transients don't bleed into
        # the policy's first step.
        self.data.qvel[:] = 0.0

        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_base_pos = self.data.body(self.base_body_id).xpos.copy()
        self.prev_action = np.zeros(self.action_dim)

        # Reset feet-air-time state. Initialize `prev_contact` from the settled pose
        # so the very first control step doesn't fire a spurious "landing" event.
        self.foot_airtime[:] = 0.0
        self.foot_prev_contact = self._get_foot_contacts()

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """Execute one control step.

        - Torque limit (±100 Nm) is enforced physically by the actuator's gear=100
          combined with ctrlrange=[-1,1] clamp, so we write the action directly to
          data.ctrl without further scaling.
        - Velocity limit (±joint_velocity_limit) is enforced as motor saturation:
          at each sim substep, if a joint is already at its speed limit and the
          commanded torque would push it further, that torque is zeroed. Models a
          real motor's back-EMF: you can't spin faster than the motor's no-load speed.
        """
        action = np.array(action, dtype=np.float32)
        # Safety clip against policies that occasionally output slightly > 1.
        action = np.clip(action, -1.0, 1.0)
        self._last_action = action.copy()

        action_repeat = self.config["ROBOT"]["action_repeat"]
        vel_limit = self.config["ROBOT"]["joint_velocity_limit"]

        for _ in range(action_repeat):
            ctrl = action.copy()
            # Motor saturation: zero torque that would push an already-maxed-out joint.
            for act_i, dof_i in enumerate(self.actuator_dof_indices):
                if dof_i < 0:
                    continue
                qv = self.data.qvel[dof_i]
                if qv > vel_limit and ctrl[act_i] > 0.0:
                    ctrl[act_i] = 0.0
                elif qv < -vel_limit and ctrl[act_i] < 0.0:
                    ctrl[act_i] = 0.0
            np.copyto(self.data.ctrl, ctrl)
            mujoco.mj_step(self.model, self.data)

        obs = self._get_observation()
        reward = self._get_reward()

        terminated, reason = self._check_termination()
        truncated = False

        # Terminal fall penalty. Applied at the terminating step so the Bellman
        # backup carries the signal back through the trajectory.
        if terminated and reason == "fall":
            reward -= self.config["REWARD"]["fall_penalty"]

        info = {
            "episode_reward": self.episode_reward,
            "step": self.step_count,
            "termination_reason": reason,
            "forward_speed_mps": float(getattr(self, "_last_v_along", 0.0)),
            "path_deviation": self.path_tracker.get_deviation(
                self.data.body(self.base_body_id).xpos[0],
                self.data.body(self.base_body_id).xpos[1],
            ),
        }

        self.episode_reward += reward
        self.step_count += 1
        self.prev_action = action.copy()
        self.prev_base_pos = self.data.body(self.base_body_id).xpos.copy()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            # Placeholder: MuJoCo viewer would go here
            pass
        elif self.render_mode == "rgb_array":
            # Return RGB array
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        """Close environment"""
        if self.viewer is not None:
            self.viewer.close()


def create_env(config, num_envs=1, **kwargs):
    """Create vectorized environment"""
    from stable_baselines3.common.vec_env import DummyVecEnv
    import platform

    def make_env(rank):
        def _init():
            return LeggedRobotEnv(config, **kwargs)

        return _init

    # Use DummyVecEnv on Windows, SubprocVecEnv on Linux/Mac for speed
    if platform.system() == "Windows" or num_envs == 1:
        envs = DummyVecEnv([make_env(i) for i in range(max(1, num_envs))])
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    return envs
