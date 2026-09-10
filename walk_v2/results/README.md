# walk_v2/results

* `trace_mjx*.json` — deterministic control-law traces for `tools/compare_traces.py` (cross-check with the CPU arm).
* `bench_local_cpu.json` — the fixed `bench.py` on the laptop CPU (sanity reference only).
* `bench_izar_v100_3144676.json`, `bench_izar_solver_3144783_*.json` — **pre-fix `bench_env`**: the env
  steps/s in these include a recompile inside the timed call (~600 ms/step floor, ~15× low). The PPO
  iteration numbers in `bench_izar_v100_3144676.json` are valid (fixed rollout length).
* `bench_izar_solver_3144836_*.json` — caps 1×4 / 2×4 / 4×4 with the fixed bench; fast but the plant is
  unstable at these caps (golden replay fails) — not usable settings.
* `bench3_izar_solver_3144851_*.json` — the fixed bench, caps 100×50 … 4×4, env + PPO iteration at 2048 envs.
* `profile_step_izar_*.json` — bare MJX physics per solver variant (`tools/profile_step.py`); the
  `_planar_` file is the held-stance drift check (300 ticks), the others fall (drift not meaningful there).
