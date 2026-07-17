"""Curriculum stages — the API per-experiment ./curriculum.py modules author against.

A curriculum is an ordered list of STAGES; each stage pairs a TRIGGER (when, mapped to
a progress fraction p in [0,1]) with an EFFECT (a callable applied to the live envs).
The StageCallback evaluates triggers once per rollout, calls effects through a VecEnv
proxy (so SubprocVecEnv workers get them too), and persists progress to
<run>/curriculum.json so --resume can restore it.

    from framework.curriculum import Stage, Steps, lerp

    def stages(cfg):
        return [Stage("cmd_ramp", trigger=Steps(0, cfg.curriculum_steps),
                      effect=lambda env, p: env.set_cmd_vx_frac(lerp(0.3, 0.6, p)))]
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback


def lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, p))


@dataclass
class Steps:
    """Trigger: progress ramps linearly from 0 at `start` timesteps to 1 at `end`."""
    start: int
    end: int

    def progress(self, num_timesteps: int) -> float:
        if self.end <= self.start:
            return 1.0
        return max(0.0, min(1.0, (num_timesteps - self.start) / (self.end - self.start)))


@dataclass
class Stage:
    name: str
    trigger: Steps
    effect: Callable        # effect(env_proxy, p) — env_proxy forwards setters to every worker


class _VecEnvProxy:
    """Makes `env.set_cmd_vx_frac(v)` inside an effect reach every SubprocVecEnv worker."""
    def __init__(self, venv):
        self._venv = venv

    def __getattr__(self, name):
        def call(*args):
            self._venv.env_method(name, *args)
        return call


class StageCallback(BaseCallback):
    """Generic curriculum driver for module-defined stages (framework counterpart of the
    hand-written CurriculumCallback/SprintCurriculumCallback in rl/train.py)."""
    def __init__(self, stages: list[Stage], run_dir=None):
        super().__init__()
        self.stages = stages
        self.run_dir = run_dir
        self._last: dict[str, float] = {}

    def _on_rollout_start(self) -> None:
        proxy = _VecEnvProxy(self.training_env)
        changed = False
        for st in self.stages:
            p = st.trigger.progress(self.num_timesteps)
            self.logger.record(f"curriculum/{st.name}", p)
            if self._last.get(st.name) == p:
                continue
            self._last[st.name] = p
            st.effect(proxy, p)
            changed = True
        if changed and self.run_dir:
            try:
                (Path(self.run_dir) / "curriculum.json").write_text(
                    json.dumps({"stages": self._last}))
            except OSError:
                pass

    def _on_step(self) -> bool:
        return True
