"""Do the two reflex channels still earn their place?

The v2 control law carries two feedback terms beside the latched series and the residual
(V2_CONTRACT.md, generator block): a LEARNED roll reflex on the hips (gains are latched action
dims 36:39) and a FIXED pitch reflex on the thighs (gains from the config, not in the action,
inherited from the walk_mit m3 lineage). This ablates each one on a trained policy and reports
what the greedy runner does without it.

Read the result carefully: a collapse means the policy DEPENDS on the term as trained, not that
the term is well designed. A survival means the term is removable for this policy.

    python walk_v2/tools/reflex_ablation.py --run walk_v2/runs/v2c_s2_free_dp2x_s0 \
        --checkpoint .../best_88473600.msgpack --episodes 16 --seconds 20
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

import numpy as np
import jax

from config import config_from_dict
from evaluate import rollout
from ppo import PPO
from train import make_eval_env, latest_checkpoint

CASES = [
    ("as trained", {}),
    ("fixed PITCH reflex off", dict(pitch_kp=0.0, pitch_kd=0.0, pitch_bias=0.0)),
    ("learned ROLL reflex off", dict(reflex_kp_scale=0.0, reflex_kd_scale=0.0,
                                     reflex_bias_scale=0.0)),
    ("both reflexes off", dict(pitch_kp=0.0, pitch_kd=0.0, pitch_bias=0.0,
                               reflex_kp_scale=0.0, reflex_kd_scale=0.0, reflex_bias_scale=0.0)),
    # the roll reflex is two things: feedback on measured roll, and a latched DC bias that the S2
    # runner pins at +1 (+11.5 deg) against o_hip at -1 (-8.6 deg). Split them, or "roll off"
    # measures the loss of that cancellation, not the loss of the feedback.
    ("roll FEEDBACK off, bias kept", dict(reflex_kp_scale=0.0, reflex_kd_scale=0.0)),
    ("roll BIAS off, feedback kept", dict(reflex_bias_scale=0.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="substring: run only the cases whose name contains it")
    args = ap.parse_args()

    run = Path(args.run)
    base = config_from_dict(json.loads((run / "resolved_config.json").read_text())["config"])
    ck = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run)
    out = {}
    for name, over in CASES:
        if args.only and args.only not in name:
            continue
        cfg = replace(base, **over) if over else base
        env = make_eval_env(cfg, args.episodes)
        agent = PPO(cfg, env, run, cfg.total_steps, seed=0, eval_env=None)
        agent.load(ck)
        n_max = int(round(args.seconds / env.control_dt))
        st, _ = rollout(env, agent, args.seed, n_max)
        alive = st["alive"]
        t_end = np.where(alive, args.seconds, st["t_end"])
        print("  %-26s upright %2d/%2d  mean %5.2f s (min %4.2f)  dist %5.1f m  %4.2f m/s"
              % (name, int(alive.sum()), args.episodes, t_end.mean(), t_end.min(),
                 st["dist"].mean(), st["dist"].mean() / max(t_end.mean(), 1e-6)))
        out[name] = dict(upright=int(alive.sum()), t_end=t_end.tolist(),
                         dist=st["dist"].tolist())
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
