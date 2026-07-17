"""Curriculum for m3_ft.

A curriculum is an ordered list of STAGES; each stage pairs a TRIGGER (when) with an
EFFECT (what changes on the live env). The framework evaluates triggers every rollout,
applies effects through env setters, and persists progress to curriculum.json so
`--resume` restarts mid-ramp.

m3_ft is a max-speed FINE-TUNE: the command is already pinned at the ceiling and Z is
free, so there is no command / distance ramp here (single stage, nothing scheduled).
The entropy anneal that *does* run lives in the shared PPO base (dash01_base) and is
gated on emergent stepping — not duplicated per experiment.
"""


def stages(cfg) -> list:
    # No active curriculum for m3_ft — it warm-starts an already-moving policy.
    return []


# ---------------------------------------------------------------------------
# For reference, a REAL ramp (from m2_walk) using the same API:
#
#   from framework.curriculum import Stage, Steps, lerp
#
#   def stages(cfg):
#       return [
#           Stage("cmd_ramp",
#                 trigger=Steps(0, cfg.curriculum_steps),           # over the first N steps
#                 effect=lambda env, p: env.set_cmd_vx_frac(lerp(0.3, 0.6, p))),
#       ]
#
# and the sprint distance ramp:
#
#           Stage("sprint_dist",
#                 trigger=Steps(0, cfg.sprint_curriculum_steps),
#                 effect=lambda env, p: env.set_sprint_dist(lerp(25.0, 100.0, p))),
# ---------------------------------------------------------------------------
