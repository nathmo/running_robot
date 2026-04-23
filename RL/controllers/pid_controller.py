"""PID controller utilities for the inverted pendulum task."""

from dataclasses import dataclass


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to [-180, 180]."""
    wrapped = angle_deg
    while wrapped > 180.0:
        wrapped -= 360.0
    while wrapped < -180.0:
        wrapped += 360.0
    return wrapped


def shortest_error_deg(target_deg: float, current_deg: float) -> float:
    """Signed shortest angular error target-current in [-180, 180]."""
    return wrap_angle_deg(target_deg - current_deg)


@dataclass
class PendulumPIDGains:
    kp: float = 0.06
    ki: float = 0.001
    kd: float = 0.02


class PendulumPIDController:
    """PID controller outputting normalized action in [-1, 1].

    The environment maps action directly to torque in [-1, 1] Nm.
    """

    def __init__(
        self,
        setpoint_deg: float = 90.0,
        gains: PendulumPIDGains | None = None,
        action_limit: float = 1.0,
        integral_limit: float = 600.0,
    ):
        self.setpoint_deg = float(setpoint_deg)
        self.gains = gains or PendulumPIDGains()
        self.action_limit = float(action_limit)
        self.integral_limit = float(integral_limit)

        self.integral_error = 0.0

    def reset(self) -> None:
        self.integral_error = 0.0

    def compute(self, angle_deg: float, angular_velocity_deg_s: float, dt: float) -> float:
        """Compute torque command in normalized action units [-1, 1]."""
        dt = max(float(dt), 1e-6)

        error = shortest_error_deg(self.setpoint_deg, float(angle_deg))
        self.integral_error += error * dt
        if self.integral_error > self.integral_limit:
            self.integral_error = self.integral_limit
        elif self.integral_error < -self.integral_limit:
            self.integral_error = -self.integral_limit

        # d(error)/dt = -angular_velocity for fixed setpoint.
        derivative_error = -float(angular_velocity_deg_s)

        u = (
            self.gains.kp * error
            + self.gains.ki * self.integral_error
            + self.gains.kd * derivative_error
        )

        if u > self.action_limit:
            u = self.action_limit
        elif u < -self.action_limit:
            u = -self.action_limit

        return float(u)
