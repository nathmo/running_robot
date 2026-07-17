"""M0 acceptance gate: every rl/config.py preset, re-encoded as an experiment under
experiments/presets/, must compile to a Config that is FIELD-IDENTICAL to
get_config(<preset>). This is what makes the experiments/ schema a safe replacement
for the preset factories: same inputs, provably same trainer configuration.
"""
from dataclasses import fields
from pathlib import Path

import pytest

from framework.compile import compile_experiment
from framework.loader import load_experiment
from rl.config import PRESETS, get_config

ROOT = Path(__file__).resolve().parent.parent
ENCODINGS = ROOT / "experiments" / "presets"
LIB = ROOT / "experiments" / "_lib" / "bases"


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_preset_encoding_matches(preset):
    path = ENCODINGS / f"{preset}.yaml"
    assert path.exists(), f"missing encoding {path}"
    exp = load_experiment(path, LIB)
    compiled = compile_experiment(exp)
    want = get_config(preset)
    got = compiled.config
    diffs = []
    for f in fields(want):
        a, b = getattr(want, f.name), getattr(got, f.name)
        if a != b:
            diffs.append(f"{f.name}: preset={a!r} compiled={b!r}")
    assert not diffs, f"{preset}: compiled Config differs from get_config():\n  " + "\n  ".join(diffs)
    assert compiled.name == preset


def test_no_stray_encodings():
    names = {p.stem for p in ENCODINGS.glob("*.yaml")}
    assert names == set(PRESETS), (
        f"encodings vs PRESETS mismatch: extra={sorted(names - set(PRESETS))}, "
        f"missing={sorted(set(PRESETS) - names)}")
