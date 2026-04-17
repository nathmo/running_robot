"""
Path definitions and tracking for curved terrain navigation
"""

import numpy as np
from typing import Tuple, Dict


class Path:
    """Base class for paths the robot follows"""

    def get_position(self, t: float) -> Tuple[float, float]:
        """Get (x, y) position at parameter t in [0, 1]"""
        raise NotImplementedError

    def get_heading(self, t: float) -> float:
        """Get desired heading angle (radians) at parameter t"""
        raise NotImplementedError

    def get_progress(self, x: float, y: float) -> float:
        """Get progress along path (0 to 1) for position (x, y)"""
        raise NotImplementedError

    def get_closest_point(self, x: float, y: float) -> Tuple[float, float, float]:
        """Get closest point on path and progress parameter"""
        raise NotImplementedError


class StraightPath(Path):
    """Infinite straight line along a fixed direction. Use for straight running."""

    def __init__(self, start=(0.0, 0.0), direction=(1.0, 0.0), length=1000.0):
        dx, dy = direction
        norm = np.hypot(dx, dy)
        if norm < 1e-9:
            raise ValueError("StraightPath direction must be non-zero")
        self.start_x, self.start_y = start
        self.dir_x, self.dir_y = dx / norm, dy / norm
        self.length = length

    def get_position(self, t: float) -> Tuple[float, float]:
        x = self.start_x + self.dir_x * t * self.length
        y = self.start_y + self.dir_y * t * self.length
        return x, y

    def get_heading(self, t: float) -> float:
        return float(np.arctan2(self.dir_y, self.dir_x))

    def get_closest_point(self, x: float, y: float) -> Tuple[float, float, float]:
        dx = x - self.start_x
        dy = y - self.start_y
        s = dx * self.dir_x + dy * self.dir_y  # signed distance along direction (m)
        t = np.clip(s / self.length, 0.0, 1.0)
        cx = self.start_x + self.dir_x * t * self.length
        cy = self.start_y + self.dir_y * t * self.length
        return cx, cy, float(t)


class CircularPath(Path):
    """Circular path with specified radius"""

    def __init__(self, center_x=0.0, center_y=0.0, radius=3.0, start_angle=0.0):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.start_angle = start_angle

    def get_position(self, t: float) -> Tuple[float, float]:
        """t in [0,1] maps to full circle"""
        angle = self.start_angle + t * 2 * np.pi
        x = self.center_x + self.radius * np.cos(angle)
        y = self.center_y + self.radius * np.sin(angle)
        return x, y

    def get_heading(self, t: float) -> float:
        """Heading is tangent to circle"""
        angle = self.start_angle + t * 2 * np.pi
        # Tangent direction (perpendicular to radius)
        heading = angle + np.pi / 2
        return heading

    def get_closest_point(self, x: float, y: float) -> Tuple[float, float, float]:
        """Find closest point on circle"""
        dx = x - self.center_x
        dy = y - self.center_y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1e-6:
            # At center, return arbitrary point
            closest_x = self.center_x + self.radius
            closest_y = self.center_y
            angle = 0
        else:
            angle = np.arctan2(dy, dx)
            closest_x = self.center_x + self.radius * np.cos(angle)
            closest_y = self.center_y + self.radius * np.sin(angle)

        # Convert angle to t parameter [0,1]
        angle_normalized = angle - self.start_angle
        t = angle_normalized / (2 * np.pi)
        t = t % 1.0  # Wrap to [0,1]

        return closest_x, closest_y, t


class SineWavePath(Path):
    """Sinusoidal path along x-axis"""

    def __init__(
        self, center_y=0.0, amplitude=1.0, frequency=2.0, length=20.0, start_x=0.0
    ):
        self.center_y = center_y
        self.amplitude = amplitude
        self.frequency = frequency
        self.length = length
        self.start_x = start_x

    def get_position(self, t: float) -> Tuple[float, float]:
        """t in [0,1] maps along the sine wave"""
        x = self.start_x + t * self.length
        y = (
            self.center_y
            + self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        )
        return x, y

    def get_heading(self, t: float) -> float:
        """Heading aligned with sine wave slope"""
        # Derivative: dy/dx = amplitude * cos(2*pi*freq*t) * 2*pi*freq
        dydx = (
            self.amplitude
            * np.cos(2 * np.pi * self.frequency * t)
            * 2
            * np.pi
            * self.frequency
        )
        heading = np.arctan(dydx)
        return heading

    def get_closest_point(self, x: float, y: float) -> Tuple[float, float, float]:
        """Find closest point on sine wave (approximate)"""
        # Sample path and find closest
        t_samples = np.linspace(0, 1, 1000)
        min_dist = float("inf")
        best_t = 0

        for t in t_samples:
            px, py = self.get_position(t)
            dist = (x - px) ** 2 + (y - py) ** 2
            if dist < min_dist:
                min_dist = dist
                best_t = t

        closest_x, closest_y = self.get_position(best_t)
        return closest_x, closest_y, best_t


class SpiralPath(Path):
    """Outward or inward spiral path"""

    def __init__(
        self, center_x=0.0, center_y=0.0, start_radius=1.0, end_radius=10.0
    ):
        self.center_x = center_x
        self.center_y = center_y
        self.start_radius = start_radius
        self.end_radius = end_radius

    def get_position(self, t: float) -> Tuple[float, float]:
        """t in [0,1] maps along spiral"""
        angle = t * 4 * np.pi  # 2 full rotations
        radius = self.start_radius + (self.end_radius - self.start_radius) * t
        x = self.center_x + radius * np.cos(angle)
        y = self.center_y + radius * np.sin(angle)
        return x, y

    def get_heading(self, t: float) -> float:
        """Heading follows spiral tangent"""
        angle = t * 4 * np.pi
        # Approximate tangent
        dt = 0.001
        t_next = np.clip(t + dt, 0, 1)
        x1, y1 = self.get_position(t)
        x2, y2 = self.get_position(t_next)
        heading = np.arctan2(y2 - y1, x2 - x1)
        return heading

    def get_closest_point(self, x: float, y: float) -> Tuple[float, float, float]:
        """Find closest point on spiral (approximate)"""
        t_samples = np.linspace(0, 1, 1000)
        min_dist = float("inf")
        best_t = 0

        for t in t_samples:
            px, py = self.get_position(t)
            dist = (x - px) ** 2 + (y - py) ** 2
            if dist < min_dist:
                min_dist = dist
                best_t = t

        closest_x, closest_y = self.get_position(best_t)
        return closest_x, closest_y, best_t


class PathTracker:
    """Track robot progress along a path"""

    def __init__(self, path: Path):
        self.path = path

    def get_speed_along_path(
        self, prev_pos: Tuple[float, float], curr_pos: Tuple[float, float]
    ) -> float:
        """Compute forward speed along path direction"""
        _, _, t_prev = self.path.get_closest_point(prev_pos[0], prev_pos[1])
        _, _, t_curr = self.path.get_closest_point(curr_pos[0], curr_pos[1])

        # Handle wraparound (e.g., circles)
        dt = t_curr - t_prev
        if dt < -0.5:
            dt += 1.0
        elif dt > 0.5:
            dt -= 1.0

        return dt

    def get_deviation(self, x: float, y: float) -> float:
        """Distance from path"""
        closest_x, closest_y, _ = self.path.get_closest_point(x, y)
        deviation = np.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
        return deviation

    def get_progress(self, x: float, y: float) -> float:
        """Progress parameter [0,1]"""
        _, _, progress = self.path.get_closest_point(x, y)
        return progress


def create_random_path(path_type="circle", seed=None):
    """Create a random path of specified type"""
    if seed is not None:
        np.random.seed(seed)

    if path_type == "circle":
        radius = np.random.choice([2.0, 3.0, 5.0, 10.0])
        return CircularPath(radius=radius)

    elif path_type == "sine":
        amplitude = np.random.uniform(0.5, 2.0)
        frequency = np.random.uniform(1.0, 3.0)
        return SineWavePath(amplitude=amplitude, frequency=frequency)

    elif path_type == "spiral":
        return SpiralPath(start_radius=1.0, end_radius=np.random.uniform(5.0, 15.0))

    else:
        raise ValueError(f"Unknown path type: {path_type}")


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create different paths
    circle = CircularPath(radius=5.0)
    sine = SineWavePath(amplitude=1.0, frequency=2.0)
    spiral = SpiralPath(start_radius=1.0, end_radius=8.0)

    paths = {"Circle": circle, "Sine Wave": sine, "Spiral": spiral}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for (name, path), ax in zip(paths.items(), axes):
        t_samples = np.linspace(0, 1, 100)
        xs, ys = zip(*[path.get_position(t) for t in t_samples])

        ax.plot(xs, ys, "b-", linewidth=2, label="Path")
        ax.scatter([xs[0]], [ys[0]], c="g", s=100, label="Start", zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], c="r", s=100, label="End", zorder=5)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Test tracking
    tracker = PathTracker(circle)
    print(f"Deviation at (5.0, 0.0): {tracker.get_deviation(6.0, 0.0):.3f} m")
    print(f"Progress at (5.0, 0.0): {tracker.get_progress(5.0, 0.0):.3f}")
