import math
import gym
import numpy as np


class PendulumEnv(gym.Env):
    """Simple continuous inverted pendulum environment.

    Observation: [angle_revs, angular_velocity_revs_per_s, last_torque]
    Action: scalar torque (Nm) in range [-max_torque, max_torque]
    """

    metadata = {"render_modes": []}

    def __init__(self, dt=1/50.0, max_torque=1.0, mass=0.5, length=0.5, damping=0.05,
                 gravity=9.81, max_episode_seconds=20.0, include_torque_in_obs=True,
                 start_angle_range=0.1):
        super().__init__()
        self.dt = float(dt)
        self.max_torque = float(max_torque)
        self.m = float(mass)
        self.l = float(length)
        self.b = float(damping)
        self.g = float(gravity)
        self.include_torque = include_torque_in_obs

        self.max_episode_steps = int(max_episode_seconds / self.dt)

        # action: torque
        self.action_space = gym.spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)

        # observation: angle (revolutions), angular velocity (rev/s), last torque
        obs_high = np.array([100.0, 100.0, self.max_torque], dtype=np.float32)
        obs_low = -obs_high
        if not self.include_torque:
            obs_high = obs_high[:2]
            obs_low = obs_low[:2]
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.start_angle_range = float(start_angle_range)
        self._reset_state()

    def _reset_state(self):
        # state in radians
        self.theta = (np.random.uniform(-self.start_angle_range, self.start_angle_range) * 2.0 * math.pi)
        # random initial velocity small
        self.theta_dot = np.random.uniform(-1.0, 1.0)
        self.last_torque = 0.0
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        angle_revs = self.theta / (2.0 * math.pi)
        vel_revs = self.theta_dot / (2.0 * math.pi)
        if self.include_torque:
            return np.array([angle_revs, vel_revs, float(self.last_torque)], dtype=np.float32)
        else:
            return np.array([angle_revs, vel_revs], dtype=np.float32)

    def step(self, action):
        u = float(np.clip(action, -self.max_torque, self.max_torque).reshape(-1)[0])
        self.last_torque = u

        # simple pendulum dynamics: theta_dd = (u - m*g*l*sin(theta) - b*theta_dot) / (m*l^2)
        theta_dd = (u - self.m * self.g * self.l * math.sin(self.theta) - self.b * self.theta_dot) / (self.m * (self.l ** 2))

        # integrate
        self.theta_dot += theta_dd * self.dt
        self.theta += self.theta_dot * self.dt

        self.step_count += 1

        obs = self._get_obs()

        # reward shaping: keep upright (theta ~ 0), minimize velocity and torque
        angle_err = ((self.theta + math.pi) % (2.0 * math.pi)) - math.pi
        angle_err_revs = angle_err / (2.0 * math.pi)
        vel_revs = self.theta_dot / (2.0 * math.pi)

        # weights chosen to encourage fast stabilization with low torque
        w_angle = 1.0
        w_vel = 0.1
        w_torque = 0.01

        reward = - (w_angle * (angle_err_revs ** 2) + w_vel * (vel_revs ** 2) + w_torque * (u ** 2))

        done = False
        info = {}

        if self.step_count >= self.max_episode_steps:
            done = True

        return obs, float(reward), done, False, info

    def render(self):
        pass

    def close(self):
        pass
