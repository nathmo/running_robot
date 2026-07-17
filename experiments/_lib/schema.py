"""Compatibility shim — the schema moved to framework/schema.py (v1.1).

Kept so `from _lib.schema import ...` (the typed-authoring form used by
experiments/*/experiment.py, with experiments/ on sys.path) keeps working.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]      # repo root (…/experiments/_lib/schema.py)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from framework.schema import *                    # noqa: F401,F403,E402
from framework.schema import (                    # noqa: F401,E402  (explicit re-exports)
    Axis, AXIS_ORDER, Bind, Backend, Calibration, Command, CommandMode, Controller,
    ControllerKind, CurriculumSpec, Deploy, DeployTarget, Device, Dof, DofFree, DofLock,
    DofMode, DofRail, DofSoftLimit, DomainRand, Drives, Entropy, Experiment, Locked,
    MirrorPair, Observation, PID, PPO, PatternGen, Phase, PhaseMode, Plant, Policy,
    PositionPD, Reflex, ReflexInput, RewardSpec, Run, Safety, SafetyEvent,
    SpringDamper, Sprint, Torque,
)
