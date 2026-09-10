import csv, sys
path = sys.argv[1]; every = float(sys.argv[2]) if len(sys.argv) > 2 else 1e6
rows = [r for r in csv.DictReader(open(path)) if r.get("time/total_timesteps")]
cols = list(rows[0].keys())
print("COLUMNS:", ", ".join(cols))
want = ["rollout/ep_len_mean", "rollout/ep_rew_mean", "curriculum/pitch_assist", "curriculum/dr_scale", "curriculum/sprint_dist_m",
        "train/std", "train/ent_coef", "train/approx_kl", "train/clip_fraction", "train/explained_variance", "train/sym_loss"]
# fuzzy-resolve the rest by suffix
def find(sfx):
    for c in cols:
        if c.endswith(sfx): return c
    return None
extra = [find(s) for s in ["fwd_speed", "freq_hz_median", "freq_lo_rail", "freq_hi_rail", "res_sat", "vel_rmse", "swing_frac_min",
                            "residual", "spec_cycle", "thermal", "height", "yaw_rate", "lane", "duty_sym", "duty_min", "term_low", "term_tip", "term_ws", "term_floor", "finish", "fall_frac", "ep_len_greedy"]]
extra = [e for e in extra if e]
keys = [c for c in want if c in cols] + extra
short = [k.split("/")[-1][:12] for k in keys]
print("step_M " + " ".join(f"{s:>12}" for s in short))
nxt = 0.0
for r in rows:
    t = float(r["time/total_timesteps"])
    if t >= nxt:
        vals = []
        for k in keys:
            v = r.get(k, "")
            try: vals.append(f"{float(v):12.4g}")
            except: vals.append(f"{'':>12}")
        print(f"{t/1e6:6.1f} " + " ".join(vals))
        nxt += every
