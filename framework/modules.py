"""Load per-experiment override modules (./reward.py, ./observation.py, ...).

Resolution rule (experiments/README.md): './x.py' is relative to the experiment folder
(cfg.experiment_dir); anything else is a path relative to the repo root (library files
like 'experiments/_lib/rewards/gait_speed_v3.py').
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def resolve(module_path: str, exp_dir: str = "") -> Path:
    p = Path(module_path)
    if module_path.startswith("./") or module_path.startswith(".\\"):
        if not exp_dir:
            raise FileNotFoundError(f"'{module_path}' is experiment-local but no experiment_dir is set")
        p = Path(exp_dir) / module_path[2:]
    if not p.exists():
        raise FileNotFoundError(f"module '{module_path}' not found at {p}")
    return p


def load_module(module_path: str, exp_dir: str = ""):
    p = resolve(module_path, exp_dir)
    name = f"_expmod_{abs(hash(str(p.resolve())))}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_callable(module_path: str, exp_dir: str, fn_name: str):
    mod = load_module(module_path, exp_dir)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise AttributeError(f"module '{module_path}' does not define {fn_name}(...)")
    return fn
