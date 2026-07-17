"""Golden-trace guard: the env refactor (module-injection seams) must not change the
built-in reward/termination/observation behavior by even a bit. Traces were recorded
with tests/capture_golden.py BEFORE the refactor; this replays the same seeded action
sequences and compares exactly.
"""
import json
from pathlib import Path

import pytest

from tests.capture_golden import GOLDEN_DIR, PRESETS, trace


@pytest.mark.parametrize("preset", PRESETS)
def test_golden_trace(preset):
    ref_path = GOLDEN_DIR / f"{preset}.json"
    assert ref_path.exists(), f"golden trace missing — run `python -m tests.capture_golden` " \
                              f"from a known-good checkout first ({ref_path})"
    ref = json.loads(ref_path.read_text())
    now = trace(preset)
    assert now["terminated"] == ref["terminated"], f"{preset}: termination pattern changed"
    assert now["truncated"] == ref["truncated"], f"{preset}: truncation pattern changed"
    for i, (a, b) in enumerate(zip(now["reward"], ref["reward"])):
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), f"{preset} step {i}: reward {a} != {b}"
    for i, (a, b) in enumerate(zip(now["obs_sum"], ref["obs_sum"])):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9), f"{preset} step {i}: obs changed"
    for i, (ta, tb) in enumerate(zip(now["terms"], ref["terms"])):
        assert set(ta) == set(tb), f"{preset} step {i}: term keys changed"
        for k in ta:
            assert ta[k] == pytest.approx(tb[k], rel=1e-12, abs=1e-12), \
                f"{preset} step {i}: term {k}: {ta[k]} != {tb[k]}"
