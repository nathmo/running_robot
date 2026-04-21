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

        Convention: 0° = horizontal, 90° = upright, 270° = hanging down.
        We normalize to [-1, 1] such that 90° (upright) → 0.0.
        """
        # Wrap angle to [0, 360)
        angle_wrapped = angle_deg % 360.0

        # Map [0, 360) to [-1, 1] with 90° at 0.0
        # 0° → -1, 90° → 0, 180° → 1, 270° → 0 (wraps around)
        # Actually, simpler: offset by 90, then map [-90, 270] to [-1, 1]
        angle_offset = angle_wrapped - 90.0  # -90 to 270
        if angle_offset > 180.0:
            angle_offset -= 360.0  # Wrap to [-180, 180]

        # Normalize [-180, 180] to [-1, 1]
        angle_norm = np.clip(angle_offset / 180.0, -1.0, 1.0)
        return float(angle_norm)

    def _normalize_angular_velocity(self, ang_vel_deg_per_s):
        """Normalize angular velocity (deg/s) to [-1, 1].

        Max realistic velocity is ~360 deg/s, so we clip to that range.
        """
        max_ang_vel = 360.0  # deg/s
        ang_vel_norm = np.clip(ang_vel_deg_per_s / max_ang_vel, -1.0, 1.0)
        return float(ang_vel_norm)

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
        """Compute reward for this step.

        Terms:
          -angle_penalty: distance from upright (90°), scaled to [0, 1]
          -velocity_penalty: magnitude of angular velocity
          -effort_penalty: magnitude of control torque
        """
        r = self.config["REWARD"]

        joint_idx = self.model.jnt_dofadr[self.joint_id]
        angle_deg = float(self.data.qpos[joint_idx])
        ang_vel_deg_s = float(self.data.qvel[joint_idx])
        control_effort = float(np.abs(self.data.ctrl[0]))

        # Angle penalty: distance from 90° (upright)
        angle_error = abs(angle_deg - 90.0)
        # Normalize to [0, 1]: 0° error → 0, 180° error → 1
        angle_penalty = min(angle_error / 180.0, 1.0) * r.get("angle_weight", 1.0)

        # Velocity penalty: penalize high angular velocity
        vel_penalty = abs(ang_vel_deg_s) / 360.0 * r.get("velocity_weight", 0.1)

        # Control effort penalty: penalize large torques
        effort_penalty = control_effort * r.get("effort_weight", 0.01)

        # Bonus for being upright (within ±10°)
        upright_bonus = 0.0
        if abs(angle_deg - 90.0) <= 10.0:
            upright_bonus = r.get("upright_bonus", 0.1)

        total = upright_bonus - angle_penalty - vel_penalty - effort_penalty
        return float(total)

    def _check_termination(self):
        """Check if episode should terminate.

        Terminate if:
          1. Angle leaves [80°, 100°] band (not within ±10° of upright)
          2. Step count exceeds max
        """
        joint_idx = self.model.jnt_dofadr[self.joint_id]
        angle_deg = float(self.data.qpos[joint_idx])

        # Check upright band
        if 80.0 <= angle_deg <= 100.0:
            self.upright_time += self.config["ROBOT"]["control_dt"]
        else:
            self.upright_time = 0.0

        # Terminate if fell out of band
        if angle_deg < 80.0 or angle_deg > 100.0:
            return True, "fell"

        # Terminate if max steps reached
        if self.step_count >= self.config["RL"]["max_episode_steps"]:
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
