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

TWO GENERATIONS
---------------
Version 1 is the walk_mit bundle: 200 Hz, per-step Fourier gait, a velocity/yaw command channel,
`export_policy.py`. Version 2 is the walk_v2 (DASH-01 Walker v2) bundle: 100 Hz, a 44-dim gait
spec LATCHED at each clock wrap plus a 6-dim per-tick residual, and a task channel that is a
run/stop flag and a distance countdown rather than a velocity. `walk_v2/export.py` writes it.

They are different control laws with different runtimes (`controller.py` vs `controller_v2.py`),
so this class refuses to guess: `bundle.version` says which, and everything downstream branches on
it. What it DOES do is expose one vocabulary for the things both generations have -- the joint
band, the torque ceiling, the velocity limit, the stance -- so the safety governor, the joint map
and the arming checks are written once. Those aliases are derived, and `derived_keys` names them.

FORMAT
------
`np.savez` with float32/float64 arrays plus one JSON blob under key "meta". Loadable by numpy
alone -- no torch, no pickle, no mujoco, and nothing that executes code on load (`allow_pickle`
stays False, which also means a bundle cannot be a code-execution vector on the robot).
"""
import json

import numpy as np

BUNDLE_VERSION = 1                     # what `save` writes; v2 bundles come from walk_v2/export.py
SUPPORTED_VERSIONS = (1, 2)

# Keys every bundle must carry. Checked on load: a bundle produced by an older exporter must fail
# loudly here rather than half-configure a robot.
_NETS = ("est_w0", "est_b0", "est_w1", "est_b1", "est_w2", "est_b2",
         "pi_w0", "pi_b0", "pi_w1", "pi_b1", "act_w", "act_b", "obs_mean", "obs_var")
REQUIRED_ARRAYS = _NETS + (
    "nominal_ctrl", "default_motor_pos", "ctrl_lo", "ctrl_hi",
    "motor_vel_limit", "forcerange", "imp_kp_base", "imp_kd_base", "imp_leg_ix",
    "hist_idx",
)
REQUIRED_ARRAYS_V2 = _NETS + (
    "nominal_ctrl", "default_motor_pos", "q_lo", "q_hi",
    "motor_vel_limit", "forcerange", "drive_kp", "drive_kd", "stand_torque",
    "hist_idx", "latched_dims",
)
# meta keys the v2 runtime reads that the FIRST v2 exporter did not write. Named here so a stale
# bundle says which field is missing instead of dying inside the control law on tick 1.
REQUIRED_META_V2 = ("control_dt", "frame_dim", "history_len", "actor_dim", "action_dim",
                    "once_dim", "obs_scales", "clip_obs", "obs_eps", "gait", "spec_source",
                    "pitch_reflex_rate_lp", "motor_accel_limit", "n_harmonics", "objective",
                    "task_brake_m", "term_gravity_z", "lp_yaw_tau_s",
                    "est_hidden", "policy_hidden")


class Bundle:
    """Read-only view of a policy bundle. Arrays are numpy, scalars come from `meta`."""

    def __init__(self, arrays, meta):
        version = int(meta.get("bundle_version", -1))
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"bundle version {meta.get('bundle_version')} is not one of "
                             f"{SUPPORTED_VERSIONS} — this runtime cannot vouch for it; re-export")
        self.version = version
        required = REQUIRED_ARRAYS if version == 1 else REQUIRED_ARRAYS_V2
        missing = [k for k in required if k not in arrays]
        if missing:
            exporter = ("robot/deploy/export_policy.py" if version == 1 else "walk_v2/export.py")
            raise ValueError(f"v{version} policy bundle is missing {missing} — re-export it with "
                             f"{exporter}")
        self.a = {k: np.asarray(v) for k, v in arrays.items()}
        self.meta = dict(meta)
        self.derived_keys = ()
        if version == 2:
            self._v2_missing_meta()
            self._v2_aliases()
        self._check_shapes()

    # ------------------------------------------------------------------ v2 compatibility
    def _v2_missing_meta(self):
        missing = [k for k in REQUIRED_META_V2 if k not in self.meta]
        if missing:
            raise ValueError(f"v2 policy bundle's meta is missing {missing} — it was written by an "
                             f"older walk_v2/export.py than this runtime; re-export")

    def _v2_aliases(self):
        """One vocabulary for the things both generations have.

        These are DERIVED, not read off the file: `ctrl_lo/hi` is the v2 joint range `q_lo/q_hi`
        (the sim clips the gait target to it before the slew limit, so it is the same band the
        governor's position clamp should enforce), and `imp_kp_base/imp_kd_base` are the plant's
        base drive gains, which is what the phase-scheduled profile multiplies. Nothing here
        changes a number; it renames one."""
        a, m = self.a, self.meta
        a.setdefault("ctrl_lo", a["q_lo"])
        a.setdefault("ctrl_hi", a["q_hi"])
        a.setdefault("imp_kp_base", np.asarray(m["gait"]["drive_kp"], float))
        a.setdefault("imp_kd_base", np.asarray(m["gait"]["drive_kd"], float))
        self.derived_keys = ("ctrl_lo", "ctrl_hi", "imp_kp_base", "imp_kd_base")
        m.setdefault("nu", int(np.asarray(a["nominal_ctrl"]).size))
        # PolicyNet reads these two by name; v2 calls them actor_dim / obs_eps
        m.setdefault("vn_epsilon", float(m["obs_eps"]))
        # v2 had no velocity command channel: its task was [run flag, distance countdown], and the
        # countdown came from ground-truth world x, which the robot can only estimate. v3 replaces
        # both dims with [commanded speed / v_max, reserved] -- a joystick fraction, no odometry.
        # `command.kind` says which generation this bundle is; the legacy cmd_* keys stay zeroed so a
        # caller that asks is told "none" rather than getting a KeyError.
        cmd = m.get("command") or {}
        m.setdefault("command", {"kind": "run_flag_distance", "v_max": 0.0})
        m.setdefault("cmd_v_fwd_trained", float(cmd.get("v_max", m.get("v_max", 0.0)) or 0.0))
        for k in ("cmd_v_back_trained", "cmd_yaw_trained"):
            m.setdefault(k, 0.0)

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
        n_actor = self.n_actor
        if self.version == 2:
            if fd * hl + int(m["once_dim"]) != n_actor:
                raise ValueError(
                    "v2 bundle: {} frames x {} + a {}-wide once-block is {}, but actor_dim says {}"
                    .format(hl, fd, m["once_dim"], fd * hl + int(m["once_dim"]), n_actor))
            for k, want in (("est_w0", (m["est_hidden"][0], n_actor)),
                            ("est_w2", (3, m["est_hidden"][1])),
                            ("pi_w0", (m["policy_hidden"][0], n_actor + 3)),
                            ("act_w", (ad, m["policy_hidden"][-1])),
                            ("obs_mean", (n_actor,)), ("obs_var", (n_actor,)),
                            ("nominal_ctrl", (nu,)), ("default_motor_pos", (nu,)),
                            ("q_lo", (nu,)), ("q_hi", (nu,)), ("hist_idx", (hl,)),
                            ("latched_dims", (ad,))):
                got = tuple(self.a[k].shape)
                if got != tuple(want):
                    raise ValueError(f"bundle array {k!r} has shape {got}, expected {tuple(want)} "
                                     f"— the bundle and its meta disagree")
            return
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
        """Width of the actor's observation -- what the estimator and the policy stack read.

        v1: the stacked history is the whole thing. v2 adds the once-block (the live latched spec,
        the task channel and the commit flag), so the exporter states it and the history alone is
        the wrong number by 47."""
        if self.version == 2:
            return int(self.meta["actor_dim"])
        return int(self.meta["frame_dim"]) * int(self.meta["history_len"])

    @property
    def control_hz(self):
        return 1.0 / float(self.meta["control_dt"])

    # ------------------------------------------------------------------ the command channel
    @property
    def command_kind(self):
        """What an operator can ASK this checkpoint for, as one word. Three answers, and they are
        different inputs rather than different units of one input:

          "velocity"  v1 (walk_mit): a forward speed and a yaw rate, fixed at arm time.
          "run_stop"  v2 sprint/stoplight: task[0] is a green light, 1 or 0, and nothing else. A
                      checkpoint trained with objective='speed' also lands here, with the flag
                      pinned -- see `has_run_flag` in the panel.
          "speed"     v2 joystick: task[0] IS the commanded speed, clip(v_cmd / v_max, .., 1). The
                      panel is a slider, 0 means walk in place, and there is no separate stop.

        The panel offers exactly one of the three, because a control that cannot reach the policy
        is worse than no control: it silently does nothing while the operator believes it did."""
        if self.version != 2:
            return "velocity"
        # the exporter states it outright ("speed_fraction" / "run_flag_distance"); the objective
        # string is the fallback for a bundle written before it did
        kind = str((self.meta.get("command") or {}).get("kind") or "")
        if kind:
            return "speed" if kind == "speed_fraction" else "run_stop"
        return "speed" if str(self.meta.get("objective")) == "joystick" else "run_stop"

    @property
    def v_max(self):
        """The speed the command channel is normalised BY (m/s), for a 'speed' bundle. 0 means the
        exporter did not write one, and the runtime refuses rather than inventing a scale."""
        return float(self.meta.get("v_max") or 0.0)

    @property
    def v_min(self):
        """The bottom of the commandable range (m/s). 0 while the trainer clips there; negative
        once walking backwards is trained, and then the slider grows a left half."""
        return float(self.meta.get("v_min") or 0.0)

    @property
    def v_trained(self):
        """(lo, hi) in m/s that this CHECKPOINT was actually commanded, which is not the same as
        what the channel spans.

        The joystick's command is drawn from a fraction band that a curriculum widens downward
        (`cmd_range_start` 0.8-1.0 -> `cmd_range` 0.0-1.0), because the warm-start parent only
        knows one speed. A checkpoint taken mid-ramp has therefore never been asked to go slowly:
        its slider still runs to 0, but 0 is off-distribution and the panel has to say so. Read
        from the checkpoint's own curriculum sidecar, never from the config -- that is the same
        mistake as `freq_lo`, which cost 0.4 rad of thigh target."""
        lo, hi = self.v_min, self.v_max
        ep = self.meta.get("env_params_at_checkpoint") or {}
        f_lo, f_hi = ep.get("cmd_lo"), ep.get("cmd_hi")
        if hi > 0.0 and f_lo is not None and f_hi is not None:
            return float(f_lo) * hi, float(f_hi) * hi
        return lo, hi

    def cfg_view(self):
        """A duck-typed stand-in for walk_mit.config.Config, holding only the fields the gait
        reconstruction reads. `gait.assemble` takes a `cfg`; on the robot that cfg is this."""
        return _CfgView(self.meta["gait_cfg"])

    def gait_params(self):
        """v2 only: the `gait_v2.GaitParams` the control law is built from."""
        if self.version != 2:
            raise ValueError("gait_params() is the v2 gait block; a v1 bundle uses cfg_view()")
        import gait_v2
        return gait_v2.GaitParams.from_meta(self.meta["gait"])


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
