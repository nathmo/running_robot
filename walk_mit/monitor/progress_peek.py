import csv, sys
cols = ["time/total_timesteps", "rollout/ep_len_mean", "rollout/ep_rew_mean", "reward_terms/fwd_speed",
        "reward_terms/spec_cycle", "reward_terms/knob", "reward_terms/thermal", "reward_terms/residual",
        "reward_terms/height", "diag/freq_hz_median", "diag/freq_lo_rail", "diag/freq_hi_rail", "diag/res_sat",
        "est/vel_rmse", "train/sym_loss", "train/std", "curriculum/swing_frac_min", "curriculum/dr_scale",
        "curriculum/pitch_assist", "curriculum/sprint_dist_m", "curriculum/ent_coef", "curriculum/log_std_clamp", "time/fps"]
for path in sys.argv[1:]:
    rows = [r for r in csv.DictReader(open(path)) if r.get("time/total_timesteps")]
    print("==", path, len(rows), "rollouts")
    n = len(rows)
    for i in sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1])):
        r = rows[i]
        out = []
        for k in cols:
            try:
                out.append("%s=%.4g" % (k.split("/")[-1], float(r[k])))
            except (KeyError, ValueError, TypeError):
                pass
        print("  " + "  ".join(out))
