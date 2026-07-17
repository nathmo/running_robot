"""Scripted steering for EVALUATION (optional).

Training samples commands from the `command:` block in experiment.yaml — NOT from this
file. This is only for `evaluate.py` rollouts / videos when you want a repeatable
command profile. Live keyboard steering uses rl/joystick.py instead.

`command(t, ctx)` returns [forward, yaw] in [-1, 1] for sim-time t (seconds).
`ctx` exposes read-only rollout state (ctx.x, ctx.vx, ...) if you want closed-loop
scripting.

m3_ft is a pure sprint-forward policy, so the eval profile just holds full forward.
"""


def command(t, ctx):
    return [1.0, 0.0]        # full forward, no yaw, whole rollout


# Example of a richer profile (stand -> accelerate -> gentle turn):
#   def command(t, ctx):
#       if t < 2.0:  return [0.0, 0.0]      # stand
#       if t < 6.0:  return [1.0, 0.0]      # accelerate straight
#       return [1.0, 0.3]                   # forward + slow left
