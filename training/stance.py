"""The one correct way to give this robot a different stance (e.g. a crouch).

Four routes were tried and all four are wrong. Keeping the recipe in one place because every one of
them produced a plausible-looking table of zeros before the pose was ever looked at:

  1. Ship a crouched keyframe as an XML variant (make_balanced_keyframe --crouch). The stance it
     carries is an equilibrium of the SPRING ankle, and every walk_fwd rung runs ankle_mode="rigid",
     which both welds the ankle at dash01.xml's lock_ankle angle and zeroes the ankle spring.
     _resettle_keyframe re-solves it onto the four-bar's fold branch: base_z 0.962 -> 0.816, toe
     441 mm ahead of the CoM, every seed dead at 0.2 s.
  2. Solve the variant against the rigid ankle instead. Same collapse -- because
  3. a BALANCED stance is not a static fixed point of the settle at all. Measured: at toe-CoM ~ 0
     the pinned-base settle always drifts (that is just an inverted pendulum being one), so there is
     nothing to converge to and every "solved" crouch was a 2 s snapshot of a drift.
  4. Set key_ctrl AFTER constructing the env and resettle again. That is a SECOND settle starting
     from the first one's output, and _settle is not idempotent: toe-CoM +29.8 mm, and the control
     that holds 12 s as a shipped plant died at 1.2 s.

What works: start from a model whose keyframe is ALREADY balanced (model/dash01_bal.xml), hand the
env a stance TARGET rather than a stance, and let it run its own _resettle_keyframe EXACTLY ONCE.
_settle is path-dependent -- the starting qpos matters as much as the ctrl, which is why the base
model has to be the balanced one and not dash01.xml (from there the same ctrl lands at toe-CoM
+19.3 mm and 0/8).

ALWAYS run acceptance() before reading any A/B built on this.
"""
import numpy as np
import mujoco

BAL = "model/dash01_bal.xml"
#: key_ctrl (cam, thigh) of the shipped balanced stance -- the control condition.
SHIPPED = (-0.0887, 0.1925)


def prepare_cfg(cfg, stance=None):
    """Suppress the constructor's resettle when we are going to do it ourselves."""
    if stance is not None:
        cfg.ankle_resettle = False
    return cfg


def apply(env, stance):
    """Re-solve the env's keyframe for stance target `(cam, thigh)`. Call ONCE, after construction,
    and only on an env built with prepare_cfg(..., stance)."""
    if stance is None:
        return env
    cam, thigh = float(stance[0]), float(stance[1])
    env.model.key_ctrl[env.key_id] = np.array([0., cam, thigh, 0., -cam, -thigh])
    env.nominal_ctrl[:] = env.model.key_ctrl[env.key_id]
    env._resettle_keyframe()
    return env


def settled(env):
    """(base_z, toe_x - com_x) of the stance the env actually reset into. Report this in every
    table: a stance that silently collapsed looks exactly like a stance that failed on its merits."""
    d = env.data
    mujoco.mj_resetDataKeyframe(env.model, d, env.key_id)
    mujoco.mj_forward(env.model, d)
    toe = float(np.mean([d.geom_xpos[g][0] for g in env.foot_gids]))
    return float(env.height_target), toe - float(d.subtree_com[0][0])


def crouch_of(base_z, ref=1.0061):
    """Metres of ride height given up versus the shipped balanced stance."""
    return ref - float(base_z)
