# `experiments/` — the experiment format (framework wired: loader + compiler + tests)

An **experiment is a folder** (or single .yaml): a declarative `experiment.yaml` that
**inherits** a library base and **overrides only what differs**, referencing reward /
observation / curriculum pieces from `_lib/` by name. When you need bespoke behaviour
on one axis, drop a local Python file into the folder — it overrides just that axis.

```
experiments/
  presets/                20 encodings of every rl/config.py preset — the parity anchors
  m3_ft/                  FULL / explicit form — every axis + all local-module escape hatches
    experiment.yaml         the 11 axes, in full
    reward.py               local editable reward (copy of the library one)
    observation.py          what the policy sees
    network.py              architecture escape hatch
    curriculum.py           staged difficulty (empty — a fine-tune)
    steering.py             scripted command profile for eval videos
    experiment.py           the SAME experiment as a typed pydantic object
  m3_ft_clean/            SIMPLE form — the SAME experiment, a few lines
  _lib/
    bases/dash01_base.yaml     plant + timing + PPO defaults (mirrors Config defaults)
    bases/dash01_speed.yaml    + the max-forward-speed command convention
    bases/dash01_fourier.yaml  + fourier controllers + symmetry + fourier PPO
    rewards/gait_speed_v3.py   the stock v3 reward (exact extraction of the built-in)
    obs/standard.py            the stock observation frame (exact extraction)
```

## Using it

```bash
# validate an experiment (schema + runtime-capability check) and see what it changes
.venv/Scripts/python.exe -m framework.validate experiments/m3_ft_clean --diff

# train it (same trainer, new entry point; --preset still works unchanged)
.venv/Scripts/python.exe -m rl.train --experiment experiments/m3_ft_clean --n-envs 6 --subproc

# evaluate/gait_probe pick up the run's resolved_config.json automatically
.venv/Scripts/python.exe -m rl.evaluate --run rl/runs/m3_ft --viewer
```

Every run now writes `resolved_config.json` (the full resolved Config — ground truth
for reproduction and diffs) next to the legacy `preset.json`.

## The machinery (framework/)

* `framework/schema.py` — the typed pydantic v1.1 schema. **`description` is required.**
  Reward/curriculum are `{module, params}` specs; `reward.params` map onto the Config
  weight fields (`w_foot_slip`, `gait_cmd_gate`, …) and are the primary sweep axes.
* `framework/loader.py` — YAML → validated `Experiment` (inherits deep-merge +
  authoring shorthands: positional `base_dof` lists, `free` / `{lock: 0}` /
  `{rail: [a,b]}`, bare reward strings).
* `framework/compile.py` — `Experiment` → `rl.config.Config`, **capability-gated**:
  what the runtime can't run yet (per-joint torque/PID mixing, `soft_limit` DOFs,
  `phase.tune: action`, GPU+fourier, …) is an explicit error, never a silent degrade.
* `framework/modules.py` / `framework/curriculum.py` — local-module loading and the
  `Stage`/`Steps`/`lerp` curriculum API.

## The guarantees (tests/)

* `tests/test_preset_parity.py` — all 20 `experiments/presets/*.yaml` compile
  **field-identical** to `get_config(<preset>)`.
* `tests/test_env_parity.py` — golden traces: the module-injection refactor left the
  built-in env behavior bit-identical.
* `tests/test_module_parity.py` — the stock library reward/obs modules are numerically
  identical to the built-in paths, so a local `./reward.py` is a safe escape hatch.

## The 11 axes (see the taxonomy doc for the full definition)

① plant · ② per-joint controller · ③ symmetry groups · ④ per-DOF restriction ·
⑤ observation · ⑥ command (+ sprint block, episode length) · ⑦ reward (module+params) ·
⑧ curriculum · ⑨ network/RL · ⑩ backend · ⑪ run/deploy/safety (+ dr block).

Axis ② `pattern_gen` carries `update: per_cycle|per_step` — coefficients latched for a whole
gait cycle (macro-step, `fourier`) vs re-emitted every 50 Hz control step (`fourier_step`).
