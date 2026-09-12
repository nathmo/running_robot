"""Precompute base height vs (pitch, roll) with the lowest toe exactly on the floor.

The bring-up randomisation drops the robot from "5-10 cm above touching", and at +-20 deg of pitch
"touching" is several centimetres away from the keyframe height -- the foot arc is ~0.3 m long, so a
tilt swings the lowest toe a long way. Solving that inside `_reset_one` is not an option (it is a
MuJoCo root-find, and the reset is jitted), so solve it here on a grid and interpolate in-env.

`bringup_probe.touch_height()` already does the solve for one (pitch, roll); this walks a grid and
writes `walk_v3/model/touch_height.npz`.

    python walk_v3/tools/make_touch_table.py --preset v3_joystick_s2
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))   # tools/ itself, for bringup_probe

from config import PRESETS
from bringup_probe import touch_height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v3_joystick_s2")
    ap.add_argument("--pitch-deg", type=float, default=25.0, help="grid half-range (train range is 20)")
    ap.add_argument("--roll-deg", type=float, default=12.0)
    ap.add_argument("--n-pitch", type=int, default=21)
    ap.add_argument("--n-roll", type=int, default=9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = PRESETS[args.preset]()
    pitches = np.linspace(-args.pitch_deg, args.pitch_deg, args.n_pitch)
    rolls = np.linspace(-args.roll_deg, args.roll_deg, args.n_roll)
    z = np.zeros((args.n_pitch, args.n_roll), np.float32)
    for i, pd in enumerate(pitches):
        for j, rd in enumerate(rolls):
            z[i, j] = touch_height(cfg, np.deg2rad(pd), np.deg2rad(rd))[0]   # (z_touch, z_key)
        print(f"[touch] pitch {pd:+6.1f} deg: z {z[i].min():.4f} .. {z[i].max():.4f} m", flush=True)

    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "model" / "touch_height.npz"
    np.savez(out, pitch_rad=np.deg2rad(pitches).astype(np.float32),
             roll_rad=np.deg2rad(rolls).astype(np.float32), z=z, model=cfg.model_path)
    print(f"[touch] wrote {out} ({args.n_pitch}x{args.n_roll}); flat-stance z = "
          f"{z[args.n_pitch // 2, args.n_roll // 2]:.4f} m")


if __name__ == "__main__":
    main()
