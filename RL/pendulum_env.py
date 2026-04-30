"""Backward-compatible wrapper for the pendulum environment.

The real implementation lives in `environment.pendulum_env`.
"""

from environment.pendulum_env import InvertedPendulumEnv, create_env

__all__ = ["InvertedPendulumEnv", "create_env"]
