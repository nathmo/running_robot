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

    Task: Keep the pendulum upright while minimizing motor effort.

    Observation matches the hardware telemetry used on the Raspberry Pi:
        [position_turns, velocity_turns_per_s, torque_nm]
    The policy outputs a torque command in Nm, clipped to [-1, 1].
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
            "angle_penalty": 0.0,
            "velocity_penalty": 0.0,
            "effort_penalty": 0.0,
            "alive_bonus": 0.0,
            "stable_bonus": 0.0,
            "success_bonus": 0.0,
        }

        self.episode_count = 0
        self.curriculum_progress = 0.0
        self.success_achieved = False
        self.first_stable_step = None
        self.left_stable_after_entry = False
        self.stable_time = 0.0

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
        # Action space: direct torque command in Nm, clipped by the env.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space: [position_turns, velocity_turns_per_s, last_torque_nm]
        self.observation_space = spaces.Box(
            low=np.array([-100.0, -100.0, -1.0], dtype=np.float32),
            high=np.array([100.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32,
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
        """Get raw hardware-style observation values."""
        # Get joint angle (degrees) and velocity
        joint_idx = self.model.jnt_dofadr[self.joint_id]

        angle_deg = float(self.data.qpos[joint_idx])  # degrees (compiler angle="degree")
        ang_vel_deg_s = float(self.data.qvel[joint_idx])  # degrees/s

        angle_turns = angle_deg / 360.0
        ang_vel_turns_s = ang_vel_deg_s / 360.0
        obs = np.array([angle_turns, ang_vel_turns_s, float(self.prev_action)], dtype=np.float32)
        return obs

    def set_curriculum_progress(self, progress):
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def _sample_start_angle_deg(self):
        start_cfg = self.config.get("START", {})
        if not start_cfg.get("randomize_start_angle", False):
            return float(start_cfg.get("default_start_angle", 90.0))

        progress = float(np.clip(self.curriculum_progress, 0.0, 1.0))
        curriculum_episodes = max(1, int(self.config.get("RL", {}).get("curriculum_episodes", 400)))
        episode_progress = min(1.0, self.episode_count / curriculum_episodes)
        progress = max(progress, episode_progress)

        initial_span = float(start_cfg.get("curriculum_initial_span_deg", 10.0))
        final_span = float(start_cfg.get("curriculum_final_span_deg", 180.0))
        span = initial_span + (final_span - initial_span) * progress

        # Near-upright starts first; then occasionally sample the full circle.
        if np.random.rand() < progress:
            return float(np.random.uniform(0.0, 360.0))
        return float((90.0 + np.random.uniform(-span, span)) % 360.0)

    def _is_within_success_band(self):
        joint_idx = self.model.jnt_dofadr[self.joint_id]
        angle_deg = float(self.data.qpos[joint_idx])
        ang_vel_deg_s = float(self.data.qvel[joint_idx])

        angle_error_turns = abs(self._shortest_distance_to_upright(angle_deg) / 360.0)
        velocity_turns_s = abs(ang_vel_deg_s / 360.0)

        r_cfg = self.config.get("RL", {})
        return (
            angle_error_turns <= float(r_cfg.get("success_angle_threshold_turns", 0.1))
            and velocity_turns_s <= float(r_cfg.get("success_velocity_threshold_turns_s", 0.1))
        )

    def _get_reward(self):
        """Compute dense shaping reward around the hardware success criteria."""
        r = self.config["REWARD"]

        joint_idx = self.model.jnt_dofadr[self.joint_id]
        angle_deg = float(self.data.qpos[joint_idx])
        ang_vel_deg_s = float(self.data.qvel[joint_idx])
        torque_nm = float(self.data.ctrl[0])

        shortest_distance = self._shortest_distance_to_upright(angle_deg)
        angle_error_turns = shortest_distance / 360.0
        ang_vel_turns_s = ang_vel_deg_s / 360.0

        angle_penalty = r.get("angle_weight", 10.0) * (angle_error_turns ** 2)
        velocity_penalty = r.get("velocity_weight", 0.5) * (ang_vel_turns_s ** 2)
        effort_penalty = r.get("effort_weight", 0.02) * (torque_nm ** 2)

        alive_bonus = r.get("alive_bonus", 0.02)
        stable_bonus = r.get("stable_bonus", 0.0) if self._is_within_success_band() else 0.0

        total = alive_bonus + stable_bonus - angle_penalty - velocity_penalty - effort_penalty

        # Store breakdown for evaluation
        self._reward_breakdown = {
            "angle_penalty": float(angle_penalty),
            "velocity_penalty": float(velocity_penalty),
            "effort_penalty": float(effort_penalty),
            "alive_bonus": float(alive_bonus),
            "stable_bonus": float(stable_bonus),
            "shortest_distance": float(shortest_distance),
            "angle": float(angle_deg),
        }

        return float(total)

    def _check_termination(self):
        """Check if episode should terminate.

        The episode runs for the full 20 seconds. Success is measured by reaching
        the upright band within 5 seconds and then never leaving it.
        """
        rl_cfg = self.config.get("RL", {})
        max_steps = int(rl_cfg.get("max_episode_steps", 1000))
        control_dt = float(self.config["ROBOT"]["control_dt"])
        success_deadline_steps = int(float(rl_cfg.get("success_deadline_seconds", 5.0)) / control_dt)

        within = self._is_within_success_band()

        if within:
            self.stable_time += control_dt
            if self.first_stable_step is None:
                self.first_stable_step = self.step_count
        else:
            if self.first_stable_step is not None:
                self.left_stable_after_entry = True
            self.stable_time = 0.0

        if self.step_count >= max_steps:
            self.success_achieved = (
                self.first_stable_step is not None
                and self.first_stable_step <= success_deadline_steps
                and within
                and not self.left_stable_after_entry
            )
            return True, "timeout"

        return False, None

    def reset(self, seed=None, options=None):
        """Reset environment.

        Start angle follows the curriculum: initially close to upright, then
        gradually expands to full-swing random starts.
        """
        super().reset(seed=seed)

        self.episode_count += 1
        start_cfg = self.config.get("START", {})
        if start_cfg.get("curriculum_enabled", False):
            curriculum_episodes = max(1, int(self.config.get("RL", {}).get("curriculum_episodes", 400)))
            self.curriculum_progress = min(1.0, self.episode_count / curriculum_episodes)

        self.data.qpos[:] = self.model.qpos0.copy()
        self.data.qvel[:] = np.zeros(self.num_dofs)

        # Set start angle
        start_angle = self._sample_start_angle_deg()

        joint_idx = self.model.jnt_dofadr[self.joint_id]
        self.data.qpos[joint_idx] = start_angle

        # Settle physics
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self.data.qvel[:] = 0.0

        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_action = 0.0
        self.stable_time = 0.0
        self.success_achieved = False
        self.first_stable_step = None
        self.left_stable_after_entry = False

        obs = self._get_observation()
        info = {
            "curriculum_progress": self.curriculum_progress,
        }

        return obs, info

    def step(self, action):
        """Execute one control step"""
        action = np.array(action, dtype=np.float32).flatten()
        action = np.clip(action[0], -1.0, 1.0)
        self.prev_action = action

        # Apply control in Nm. The XML gear=1 means ctrl maps directly to torque.
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
            "stable_time": self.stable_time,
            "success_achieved": self.success_achieved,
            "first_stable_step": self.first_stable_step,
            "reward_breakdown": self._reward_breakdown.copy(),
        }

        self.episode_reward += reward
        self.step_count += 1

        if terminated and reason == "timeout" and self.success_achieved:
            success_bonus = self.config.get("REWARD", {}).get("success_bonus", 200.0)
            reward += success_bonus
            self.episode_reward += success_bonus
            info["reward_breakdown"]["success_bonus"] = float(success_bonus)

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
