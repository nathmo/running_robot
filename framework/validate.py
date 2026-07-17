"""Preflight an experiment: schema validation + runtime-capability compile check.

    .venv/Scripts/python.exe -m framework.validate experiments/presets/m2.yaml
    .venv/Scripts/python.exe -m framework.validate experiments/m3_ft_clean --diff

Exit code 0 = the experiment is loadable AND runnable by the current runtime.
--diff prints every compiled Config field that differs from the Config() defaults, which
is the quickest way to see what an experiment actually changes.
"""
import argparse
import sys
from dataclasses import fields

from pydantic import ValidationError

from rl.config import Config

from .compile import CompileError, compile_experiment
from .loader import load_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="experiment folder or .yaml file")
    ap.add_argument("--lib", default="experiments/_lib/bases", help="library bases dir")
    ap.add_argument("--diff", action="store_true", help="print compiled fields != Config defaults")
    args = ap.parse_args()

    try:
        exp = load_experiment(args.experiment, args.lib)
    except (ValidationError, FileNotFoundError, ValueError) as e:
        print(f"[validate] SCHEMA FAIL: {args.experiment}\n{e}")
        return 1
    try:
        out = compile_experiment(exp)
    except CompileError as e:
        print(f"[validate] RUNTIME FAIL: {e}")
        return 2

    print(f"[validate] OK: '{out.name}' — {out.description}")
    print(f"[validate]     device={out.device} n_envs={out.n_envs_spec} "
          f"warm_start={out.warm_start} gate={out.sim2sim_gate}")
    if args.diff:
        base = Config()
        for f in fields(Config):
            a, b = getattr(base, f.name), getattr(out.config, f.name)
            if a != b:
                print(f"    {f.name}: {a!r} -> {b!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
