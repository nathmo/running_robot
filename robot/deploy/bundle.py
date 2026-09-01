"""The deployed-policy bundle: one .npz that fully determines the robot's control law.

WHY A BUNDLE AND NOT "load the checkpoint on the Pi"
----------------------------------------------------
A trained policy is only half of the control law. The other half lives in DashEnv: the stance the
targets are centred on, the Fourier reconstruction, the action filter, the actuation delay, the
observation scales, the history stride, the VecNormalize statistics, the per-episode impedance
base gains. Every one of those is a number the robot must reproduce EXACTLY or the policy is being
run open-loop on an observation it has never seen.

Several of them are not in the config file at all -- they are computed by DashEnv at construction
(`_resettle_keyframe` re-solves the standing keyframe and rewrites `nominal_ctrl` and
`default_motor_pos`), or restored from the run's curriculum.json (cmd_scale, stance_ratio, the
drive bandwidth). So the export runs the real env once and reads the numbers OFF THE LIVE OBJECT
rather than re-deriving them. Re-derivation is exactly how the four separate eval-restore bugs in
this project happened.

The bundle is therefore the single source of truth for deployment, and `verify_export.py` proves
it by running the numpy control law inside MuJoCo against the torch policy.

FORMAT
------
`np.savez` with float32/float64 arrays plus one JSON blob under key "meta". Loadable by numpy
alone -- no torch, no pickle, no mujoco, and nothing that executes code on load (`allow_pickle`
stays False, which also means a bundle cannot be a code-execution vector on the robot).
"""
import json

import numpy as np

BUNDLE_VERSION = 1

# Keys every bundle must carry. Checked on load: a bundle produced by an older exporter must fail
# loudly here rather than half-configure a robot.
REQUIRED_ARRAYS = (
    "est_w0", "est_b0", "est_w1", "est_b1", "est_w2", "est_b2",
    "pi_w0", "pi_b0", "pi_w1", "pi_b1", "act_w", "act_b",
    "obs_mean", "obs_var",
    "nominal_ctrl", "default_motor_pos", "ctrl_lo", "ctrl_hi",
    "motor_vel_limit", "forcerange", "imp_kp_base", "imp_kd_base", "imp_leg_ix",
    "hist_idx",
)


class Bundle:
    """Read-only view of a policy bundle. Arrays are numpy, scalars come from `meta`."""

    def __init__(self, arrays, meta):
        missing = [k for k in REQUIRED_ARRAYS if k not in arrays]
        if missing:
            raise ValueError(f"policy bundle is missing {missing} — re-export it with "
                             f"robot/deploy/export_policy.py")
        if int(meta.get("bundle_version", -1)) != BUNDLE_VERSION:
            raise ValueError(f"bundle version {meta.get('bundle_version')} != {BUNDLE_VERSION} "
                             f"— this runtime cannot vouch for it; re-export")
        self.a = {k: np.asarray(v) for k, v in arrays.items()}
        self.meta = dict(meta)
        self._check_shapes()

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in z.files if k != "meta"}
        return cls(arrays, meta)

    @staticmethod
    def save(path, arrays, meta):
        meta = dict(meta, bundle_version=BUNDLE_VERSION)
        Bundle(arrays, meta)                     # validate BEFORE writing, never ship a bad bundle
        np.savez(path, meta=np.array(json.dumps(meta, sort_keys=True)),
                 **{k: np.asarray(v) for k, v in arrays.items()})

    # ------------------------------------------------------------------ derived
    def __getitem__(self, k):
        return self.a[k]

    def __getattr__(self, k):                    # bundle.control_dt -> meta["control_dt"]
        try:
            return self.__dict__["meta"][k]
        except KeyError:
            raise AttributeError(k) from None

    def _check_shapes(self):
        m = self.meta
        nu, ad, fd, hl = m["nu"], m["action_dim"], m["frame_dim"], m["history_len"]
        n_actor = fd * hl
        checks = [
            ("est_w0", (m["est_hidden"][0], n_actor)),
            ("est_w2", (3, m["est_hidden"][1])),
            ("pi_w0", (m["policy_hidden"][0], n_actor + 3)),
            ("act_w", (ad, m["policy_hidden"][-1])),
            ("obs_mean", (n_actor,)),
            ("obs_var", (n_actor,)),
            ("nominal_ctrl", (nu,)),
            ("default_motor_pos", (nu,)),
            ("imp_kp_base", (nu,)),
            ("hist_idx", (hl,)),
        ]
        for k, want in checks:
            got = tuple(self.a[k].shape)
            if got != tuple(want):
                raise ValueError(f"bundle array {k!r} has shape {got}, expected {tuple(want)} — "
                                 f"the bundle and its meta disagree")

    @property
    def n_actor(self):
        return int(self.meta["frame_dim"]) * int(self.meta["history_len"])

    def cfg_view(self):
        """A duck-typed stand-in for walk_mit.config.Config, holding only the fields the gait
        reconstruction reads. `gait.assemble` takes a `cfg`; on the robot that cfg is this."""
        return _CfgView(self.meta["gait_cfg"])


class _CfgView:
    """Attribute access over a plain dict, so vendored walk_mit code runs unmodified."""

    __slots__ = ("_d",)

    def __init__(self, d):
        object.__setattr__(self, "_d", dict(d))

    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError:
            raise AttributeError(
                f"the gait config in this bundle has no {k!r} — the vendored gait code and the "
                f"exporter have drifted apart") from None

    def __setattr__(self, k, v):
        raise AttributeError("the deployed gait config is read-only")

    def as_dict(self):
        return dict(self._d)
