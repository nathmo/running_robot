"""Load a YAML experiment (folder or file) into a validated framework.schema.Experiment.

Resolution rules (experiments/README.md):
  * an experiment is a folder containing experiment.yaml (a direct .yaml path also works)
  * `inherits: <name>` deep-merges the file over `<lib_dir>/<name>.yaml`, following the
    chain (dash01_fourier -> dash01_base); the leaf always wins
  * authoring shorthands (positional base_dof, bare reward strings, ...) are normalized
    by the schema's before-validators — this loader only merges and validates
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from .schema import Experiment

DEFAULT_LIB = Path("experiments") / "_lib" / "bases"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be a mapping, got {type(data).__name__}")
    return data


def resolve_path(path: str | Path) -> Path:
    """Accept an experiment folder, a folder containing experiment.yaml, or a .yaml file."""
    p = Path(path)
    if p.is_dir():
        p = p / "experiment.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no experiment file at {p}")
    return p


def load_raw(path: str | Path, lib_dir: str | Path = DEFAULT_LIB) -> dict:
    """Read + resolve the `inherits` chain, returning the merged raw dict (pre-validation)."""
    p = resolve_path(path)
    merged = _read(p)
    seen: set[str] = set()
    while (parent := merged.get("inherits")) and parent not in seen:
        seen.add(parent)
        base_path = Path(lib_dir) / f"{parent}.yaml"
        if not base_path.exists():
            raise FileNotFoundError(f"{p}: inherits '{parent}' but {base_path} does not exist")
        base = _read(base_path)
        merged = _deep_merge(base, {k: v for k, v in merged.items() if k != "inherits"})
        merged["inherits"] = base.get("inherits")
    merged.pop("inherits", None)
    return merged


def load_experiment(path: str | Path, lib_dir: str | Path = DEFAULT_LIB) -> Experiment:
    """Load a YAML experiment, resolving `inherits` chains, and validate against the schema."""
    exp = Experiment.model_validate(load_raw(path, lib_dir))
    # remember where local modules (./reward.py, ./observation.py, ...) live
    exp_dir = resolve_path(path).parent
    object.__setattr__(exp, "_dir", str(exp_dir))
    return exp


def experiment_dir(exp: Experiment) -> str:
    """The folder an Experiment was loaded from ('' for pure in-memory objects)."""
    return getattr(exp, "_dir", "") or ""
