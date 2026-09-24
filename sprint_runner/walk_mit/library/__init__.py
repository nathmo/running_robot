"""Gait-library pipeline of the DASH-01 Walker v2 design (artifact §09): the gait is found OUTSIDE
RL as a solved periodic orbit; the network stabilizes it.

    cmaes.py           minimal CMA-ES (no dependency)
    fixed_point.py     Stage 0: return map P(x; theta), damped-Newton fixed points, Floquet
                       multipliers, per-cycle envelope flags
    library_solve.py   Stage 1: CMA-ES over the 17 reduced gait variables, continuation in speed,
                       writes the library JSON the env's spec_source="library" reads
    search.py          Stage 4: closed-loop CMA-ES over theta with a frozen stabilizer policy
    absorb.py          Stage 5: feedforward absorption of the phase-averaged residual into theta
"""
