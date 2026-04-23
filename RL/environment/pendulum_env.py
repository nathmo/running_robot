"""
MuJoCo-based inverted pendulum environment with Gymnasium interface
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path as PathLib


class InvertedPendulumEnv(gym.Env):
    """
    Gymnasium environment for training RL policies on an inverted pendulum.

    Task: Keep a 200g mass balanced upright (90°) on a 0.5m massless rod
    with a 1 Nm motor. Observations are angle and angular velocity, both
    normalized to [-1, 1].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config,
        urdf_path=None,
        render_mode=None,
    ):
        """
        Args:
            config: Configuration dict (from config.py)
            urdf_path: Override URDF path if needed
            render_mode: "human" or "rgb_array"
        """
        self.config = config
        self.render_mode = render_mode

        # Load MJCF and create MuJoCo model
        self.urdf_path = urdf_path or config["ROBOT"]["urdf_path"]
        self._load_model()

        # Create Gymnasium spaces
        self._setup_spaces()

        # State tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_action = 0.0

        # Time tracking for upright stability check
        self.upright_time = 0.0  # seconds spent within upright band

        # Reward breakdown tracking
        self._reward_breakdown = {
            "proximity": 0.0,
            "upright_bonus": 0.0,
            "effort_penalty": 0.0,
            "angle": 0.0,
        }

        # Rendering
        self.viewer = None

    def _load_model(self):
        """Load pendulum MJCF model"""
        urdf_path = PathLib(self.urdf_path)
        if not urdf_path.is_absolute():
            base_dir = PathLib(__file__).parent.parent
            urdf_path = base_dir / self.urdf_path

        if not urdf_path.exists():
            raise FileNotFoundError(f"MJCF not found: {urdf_path}")

        self.model = mujoco.MjModel.from_xml_path(str(urdf_path))
        self.data = mujoco.MjData(self.model)

        # Extract actuator info
        self.action_dim = self.model.nu  # Should be 1
        self.num_dofs = self.model.nv

        print(
            f"Loaded model: {self.action_dim} actuators, {self.num_dofs} DOFs"
        )

        # Find joint ID
        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint")
        if self.joint_id < 0:
            raise ValueError("Joint 'pendulum_joint' not found in model")

        # Find mass body ID (for inertia calculations)
        self.mass_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mass")
        if self.mass_body_id < 0:
            raise ValueError("Body 'mass' not found in model")

        print(f"Joint ID: {self.joint_id}, Mass body ID: {self.mass_body_id}")

    def _setup_spaces(self):
        """Define observation and action spaces"""
        # Action space: motor control [-1, 1] → [-1, 1] Nm (linear mapping)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space: [angle_normalized, angular_velocity_normalized]
        # Both normalized to [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _normalize_angle(self, angle_deg):
        """Normalize angle (degrees) to [-1, 1].

        Convention: 0° = horizontal (0.0), 90° = upright (0.5), 
                   180° = other side (1.0), 270° = hanging down (-0.5), 
                   -180° = opposite (−1.0).
        
        Maps angle → [-1, 1] such that:
          - 0° → 0.0
          - ±180° → ±1.0
          - 90° → 0.5 (upright)
          - 270° ≡ -90° → −0.5 (hanging down)
        """
        # Normalize angle to [-180, 180] range
        angle_normalized = angle_deg
        while angle_normalized > 180.0:
            angle_normalized -= 360.0
        while angle_normalized < -180.0:
            angle_normalized += 360.0
        
        # Map [-180, 180] directly to [-1, 1]
        angle_norm = angle_normalized / 180.0
        # Clip just in case (should be unnecessary)
        angle_norm = np.clip(angle_norm, -1.0, 1.0)
        return float(angle_norm)

    def _normalize_angular_velocity(self, ang_vel_deg_per_s):
        """Normalize angular velocity (deg/s) to [-1, 1].

        Max realistic velocity is ~360 deg/s, so we clip to that range.
        """
        max_ang_vel = 360.0  # deg/s
        ang_vel_norm = np.clip(ang_vel_deg_per_s / max_ang_vel, -1.0, 1.0)
        return float(ang_vel_norm)

    def _wrap_angle_deg(self, angle_deg):
        """Wrap an angle in degrees to [-180, 180]."""
        wrapped = float(angle_deg)
        while wrapped > 180.0:
            wrapped -= 360.0
        while wrapped < -180.0:
            wrapped += 360.0
        return wrapped

    def _shortest_distance_to_upright(self, angle_deg):
        """Return the shortest angular distance in degrees to the 90° upright target."""
        angle_diff = self._wrap_angle_deg(float(angle_deg) - 90.0)
        return abs(angle_diff)

    def _get_observation(self):
        """Get normalized angle and angular velocity"""
        # Get joint angle (degrees) and velocity
        joint_idx = self.model.jnt_dofadr[self.joint_id]

        angle_deg = float(self.data.qpos[joint_idx])  # degrees (compiler angle="degree")
        ang_vel_deg_s = float(self.data.qvel[joint_idx])  # degrees/s

        angle_norm = self._normalize_angle(angle_deg)
        ang_vel_norm = self._normalize_angular_velocity(ang_vel_deg_s)

        obs = np.array([angle_norm, ang_vel_norm], dtype=np.float32)
        return obs

    def _get_reward(self):
        """Compute reward for this step, tracking components for debugging.

        Reward structure:
          - Proximity bonus: reward based on distance to upright (90°)
            Ranges from 0 at furthest (270°) to proximity_scale at upright (90°)
          - Upright bonus: extra reward for being within ±10° of upright
          - Effort penalty: penalize large motor commands

        Returns: (total_reward, reward_breakdown_dict)
        """
        r = self.config["REWARD"]

        joint_idx = self.model.jnt_dofadr[self.joint_id]
        angle_deg = float(self.data.qpos[joint_idx])
        control_effort = float(np.abs(self.data.ctrl[0]))

        # Compute shortest angular distance to upright (90°), accounting for circular wrapping
        shortest_distance = self._shortest_distance_to_upright(angle_deg)

        # Proximity reward: square the term to heavily reward near-upright angles,
        # then subtract a baseline so sideways states (near 0° / 180°) are not
        # locally attractive fixed points.
        #
        # Raw proximity term:
        #   - At 270°: 0.00
        #   - At 180°: 0.25
        #   - At  90°: 1.00
        #
        # With baseline (default 0.30):
        #   - At 270°: -0.30
        #   - At 180°: -0.05
        #   - At  90°: +0.70
        proximity_scale = r.get("proximity_scale", 1.0)
        proximity_baseline = r.get("proximity_baseline", 0.30)
        raw_proximity = ((180.0 - shortest_distance) / 180.0) ** 2 * proximity_scale
        proximity_reward = raw_proximity - proximity_baseline

        # Extra bonus for being within ±10° of upright
        upright_bonus = 0.0
        if shortest_distance <= 10.0:
            upright_bonus = r.get("upright_bonus", 0.5)

        # Control effort penalty: penalize large motor commands
        effort_penalty = control_effort * r.get("effort_weight", 0.0)

        total = proximity_reward + upright_bonus - effort_penalty

        # Store breakdown for evaluation
        self._reward_breakdown = {
            "raw_proximity": float(raw_proximity),
            "proximity": float(proximity_reward),
            "upright_bonus": float(upright_bonus),
            "effort_penalty": float(effort_penalty),
            "shortest_distance": float(shortest_distance),
            "angle": float(angle_deg),
        }

        return float(total)

    def _check_termination(self):
        """Check if episode should terminate.

                Timeout-based termination with configurable horizons:
                    - Base horizon from RL.max_episode_steps
                    - Upright-zone horizon from RL.max_upright_episode_steps (defaults to 10x base)
        """
        joint_idx = self.model.jnt_dofadr[self.joint_id]
        angle_deg = float(self.data.qpos[joint_idx])
        shortest_distance = self._shortest_distance_to_upright(angle_deg)

        # Check upright band (for tracking upright time)
        if shortest_distance <= 10.0:
            self.upright_time += self.config["ROBOT"]["control_dt"]
        else:
            self.upright_time = 0.0

        # Determine if in upright zone (±20° from 90°)
        in_upright_zone = shortest_distance <= 20.0

        rl_cfg = self.config.get("RL", {})
        base_max_steps = int(rl_cfg.get("max_episode_steps", 2000))
        upright_max_steps = int(
            rl_cfg.get("max_upright_episode_steps", base_max_steps * 10)
        )

        # Timeout-based termination
        if in_upright_zone:
            max_steps = upright_max_steps
        else:
            max_steps = base_max_steps

        if self.step_count >= max_steps:
            return True, "timeout"

        return False, None

    def reset(self, seed=None, options=None):
        """Reset environment.

        Start angle is either fixed (270°, hanging down) or randomized
        based on config.
        """
        super().reset(seed=seed)

        self.data.qpos[:] = self.model.qpos0.copy()
        self.data.qvel[:] = np.zeros(self.num_dofs)

        # Set start angle
        start_config = self.config["START"]
        if start_config.get("randomize_start_angle", False):
            angle_range = start_config.get("start_angle_range", [270, 270])
            start_angle = np.random.uniform(angle_range[0], angle_range[1])
        else:
            start_angle = start_config.get("default_start_angle", 270.0)

        joint_idx = self.model.jnt_dofadr[self.joint_id]
        self.data.qpos[joint_idx] = start_angle

        # Settle physics
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self.data.qvel[:] = 0.0

        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_action = 0.0
        self.upright_time = 0.0

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """Execute one control step"""
        action = np.array(action, dtype=np.float32).flatten()
        action = np.clip(action[0], -1.0, 1.0)
        self.prev_action = action

        # Apply control: [-1, 1] maps directly to [-1, 1] Nm torque
        # (gear=1 in XML means ctrl directly becomes torque)
        self.data.ctrl[0] = action

        # Simulate one control step (action_repeat × sim substeps)
        action_repeat = self.config["ROBOT"]["action_repeat"]
        for _ in range(action_repeat):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_observation()
        reward = self._get_reward()

        terminated, reason = self._check_termination()
        truncated = False

        info = {
            "episode_reward": self.episode_reward,
            "step": self.step_count,
            "termination_reason": reason,
            "upright_time": self.upright_time,
            "reward_breakdown": self._reward_breakdown.copy(),
        }

        self.episode_reward += reward
        self.step_count += 1

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Render environment"""
        pass

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
            return InvertedPendulumEnv(config, **kwargs)

        return _init

    # Use DummyVecEnv on Windows, SubprocVecEnv on Linux/Mac for speed
    if platform.system() == "Windows" or num_envs == 1:
        envs = DummyVecEnv([make_env(i) for i in range(max(1, num_envs))])
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    return envs
