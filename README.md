# DASH-01 — bipedal robot 100 m sprint

Master-thesis project: a custom biped (DASH-01) learning to walk, stand and sprint with RL, and the
software that runs those policies on the real robot.

| folder | what |
|---|---|
| [`controller/`](controller/README.md) | everything that runs on the robot's Raspberry Pi: the web control UI ([`fixed_gait/webui/`](controller/fixed_gait/webui/README.md)), the policy deployment stack ([`deploy/`](controller/deploy/README.md)), motor/CAN tools, identification, the hardware model |
| `RLframework/` | MJX/JAX PPO training for the flat-foot DASH-01 walker (joystick-commanded). Launch on Izar with `slurm/izar_dash.sh` |
| [`BalanceRL/`](BalanceRL/README.md) | push-recovery stander that writes the MIT force-control frame directly; a side quest off `RLframework/` |
| [`sprint_runner/`](sprint_runner/README.md) | the restored `walk_mit` stack and the 3.06 m/s runner (`sprint_m3_mit_s0`), kept so it can still be run, measured and filmed. Not deployable on the current flat-foot robot |
| `dash-01CAD/` | the current robot's CAD export (MJCF + meshes) that the training plants and the web UI's digital twin are built from |

## What is deployed

The Pi runs the web UI (`controller/fixed_gait/webui/server.py`, systemd units in
`controller/fixed_gait/webui/systemd/`). Policies reach it as bundles in
`controller/deploy/bundles/`:

- `bal5_s0.npz` — the BalanceRL stander (v3 bundle).
- `dash_joy3_lr_s2_180M.npz` — the best RLframework walker.

Older bundles, the analytic in-air gait demo, the browser MJCF viewer and the URDF patch tools
were removed on 2026-09-30; they live in git history.

## Tests

```bash
python -m pytest controller/fixed_gait/webui/tests controller/deploy/tests
```

## History

Earlier stacks (`RL/`, `framework/`, `experiments/`, `orchestrator/`, `training/`, `robot/`,
`walk_v2`..`walk_v4`) were removed as each was superseded and live in git history. `walk_mit/`
survives only inside `sprint_runner/`.
