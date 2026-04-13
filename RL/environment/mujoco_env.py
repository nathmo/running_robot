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
from .paths import PathTracker, create_random_path, CircularPath


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

        print(f"Base body ID: {self.base_body_id}, Foot body IDs: {self.foot_body_ids}")

    def _setup_spaces(self):
        """Define observation and action spaces"""
        # Action space: motor commands
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        # Observation space: compute from actual model structure
        # qpos includes all joint positions (freejoint is 7D: 3 pos + 4 quat)
        # qvel includes all joint velocities (freejoint is 6D: 3 vel + 3 angular vel)
        qpos_dim = self.model.nq  # Total qpos dimension from MuJoCo
        qvel_dim = self.model.nv  # Total qvel dimension from MuJoCo

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
        """Reset to a new random path"""
        path_config = self.config["PATHS"]

        if self.randomize_path:
            radius = np.random.choice(path_config["radii"])
            self.current_path = CircularPath(radius=radius)
        else:
            # Use default path
            self.current_path = CircularPath(radius=path_config["radii"][0])

        self.path_tracker = PathTracker(self.current_path)

    def _get_observation(self):
        """Construct observation vector"""
        # Joint positions and velocities
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # Base pose and velocity
        base_pos = self.data.body(self.base_body_id).xpos.copy()
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

    def _get_reward(self):
        """Compute reward for this step"""
        reward_config = self.config["REWARD"]

        base_pos = self.data.body(self.base_body_id).xpos.copy()

        # Forward speed along path
        speed_along_path = self.path_tracker.get_speed_along_path(
            self.prev_base_pos[:2], base_pos[:2]
        )
        speed_reward = (
            speed_along_path
            * reward_config["forward_speed_weight"]
            * reward_config["forward_speed_scale"]
        )

        # Stability (penalize base rotation)
        base_quat = self.data.body(self.base_body_id).xquat.copy()
        # Quaternion to Euler angles (just use spin rate)
        base_ang_vel = self.data.body(self.base_body_id).cvel[3:]
        stability_penalty = (
            np.linalg.norm(base_ang_vel)
            * reward_config["stability_weight"]
        )

        # Action smoothness
        action_diff = np.linalg.norm(self._last_action - self.prev_action)
        smoothness_penalty = action_diff * reward_config["action_smoothness_weight"]

        # Energy efficiency
        energy_cost = (
            np.sum(np.abs(self.data.ctrl))
            * reward_config["energy_weight"]
        )

        # Track deviation
        deviation = self.path_tracker.get_deviation(base_pos[0], base_pos[1])
        deviation_penalty = deviation * reward_config["track_deviation_weight"]

        # Alive bonus
        alive_bonus = reward_config["alive_bonus"]

        total_reward = (
            speed_reward
            - stability_penalty
            - smoothness_penalty
            - energy_cost
            - deviation_penalty
            + alive_bonus
        )

        return float(total_reward)

    def _is_done(self):
        """Check if episode is done"""
        # Episode ends if base falls too low or goes too far off path
        base_pos = self.data.body(self.base_body_id).xpos.copy()

        # Fallen
        if base_pos[2] < 0.1:
            return True

        # Too far from path
        deviation = self.path_tracker.get_deviation(base_pos[0], base_pos[1])
        if deviation > 5.0:
            return True

        # Max steps exceeded
        if self.step_count >= self.config["RL"]["max_episode_steps"]:
            return True

        return False

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        # Reset path occasionally
        if self.randomize_path and np.random.rand() < 0.3:
            self._reset_path()

        # Reset to default pose
        self.data.qpos = self.model.qpos0.copy()
        self.data.qvel = np.zeros(self.num_dofs)

        # Add small random perturbation
        self.data.qpos[:self.num_dofs] += np.random.normal(0, 0.02, self.num_dofs)

        # Simulate a few steps to settle
        mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_base_pos = self.data.body(self.base_body_id).xpos.copy()
        self.prev_action = np.zeros(self.action_dim)

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """Execute one step of environment"""
        action = np.array(action, dtype=np.float32)
        self._last_action = action.copy()

        # Repeat action for multiple sim steps
        action_repeat = self.config["ROBOT"]["action_repeat"]

        for _ in range(action_repeat):
            # Scale action from [-1,1] to joint torques
            ctrl = action * self.config["ROBOT"]["torque_limit"]
            np.copyto(self.data.ctrl, ctrl)

            # Step simulation
            mujoco.mj_step(self.model, self.data)

        # Compute observations and rewards
        obs = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = False
        info = {
            "episode_reward": self.episode_reward,
            "step": self.step_count,
            "path_deviation": self.path_tracker.get_deviation(
                self.data.body(self.base_body_id).xpos[0],
                self.data.body(self.base_body_id).xpos[1],
            ),
        }

        # Update state
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
