"""The map between the MuJoCo joint frame the policy speaks and the motor frame the robot speaks.

    MODEL  actuator order, radians    [hip_roll_L, cam_L, thigh_L, hip_roll_R, cam_R, thigh_R]
    ROBOT  paths.MOTOR_NAMES order, normalized degrees
           [right.abd, right.cam, right.thigh, left.abd, left.cam, left.thigh]

    model_rad = radians(sign * norm_deg + offset_deg)

WHY THIS IS ITS OWN MODULE, AND WHY IT REFUSES TO RUN UNVERIFIED
-----------------------------------------------------------------
A sign error here is not a degraded gait. It is a joint that responds to a balance correction by
making the fall worse, at 200 N*m/rad, and nothing upstream can detect it: the policy's targets
stay in range, the drive tracks them, the telemetry looks healthy. The robot simply falls in a way
that looks like the policy being bad.

The project has already paid for this twice. `webui/fklut.py` documents that the captured zero pose
is NOT the MJCF qpos-0 pose -- the zero is taken with the leg near-extended, right at the 4-bar
dead centre, which puts it ~135 degrees of cam away from model zero -- and the dynamic-parameter
work found the right leg's map had never been mirrored at all (it must be cam -1 / thigh +1), which
left a loop residual stuck at 0.78 m until it was found.

`fklut.py` therefore verifies cam and thigh per side against a recorded workspace before it will
even draw an end-effector. This module needs the same discipline for all SIX joints, including
abduction, which fklut never mapped because it is not part of the sagittal 4-bar.

So: every joint carries its own `verified` flag, `check_ready()` is what the runner calls before
it is allowed to command anything, and the verification is a READ-ONLY procedure -- the robot is
limp and moved by hand throughout.

THE THREE THINGS THAT MUST BE CALIBRATED, AND HOW
--------------------------------------------------
  POSITION   sign + offset. Verified statically: hold the robot in the model's stance pose and
             confirm every joint reads what `default_motor_pos` says it should, then move each
             joint by hand and confirm the model angle moves the way the model says.
  VELOCITY   the drive reports ERPM, and NOTHING in this repo converts ERPM to joint speed -- the
             pole-pair x gear product has never been measured (the only place a number appears is
             a made-up constant inside canio's mock). So `erpm_per_rad_s` starts unset, and
             `fit_erpm_scale` regresses reported ERPM against the differentiated position from a
             hand-driven sweep. Until it is fitted, velocity is taken from the differentiated
             position instead, which is unambiguous but noisy.
  TORQUE     joint N*m per reported amp. The model's `motor_kt_joint` is the DATASHEET value
             (0.272 N*m/A x 8:1 = 2.176 for the AKE90-8); the measured effective figure is 82-98%
             of it because the gearbox is not ideal, and the model's own force ranges already use
             0.85. Deploying the datasheet value would overstate every torque the policy observes
             by ~15%, so `kt_efficiency` defaults to that measured 0.85 and is a config knob.
"""
import json

import numpy as np

# MuJoCo actuator order -> (hardware motor name)
MODEL_TO_MOTOR = ("left.abd", "left.cam", "left.thigh",
                  "right.abd", "right.cam", "right.thigh")
MODEL_ACTUATORS = ("hip_roll_L", "cam_L", "thigh_L", "hip_roll_R", "cam_R", "thigh_R")
# paths.MOTOR_NAMES order, repeated here so this module does not need the webui on the desktop
MOTOR_NAMES = ("right.abd", "right.cam", "right.thigh", "left.abd", "left.cam", "left.thigh")
# measured gearbox efficiency, robot/identification 2026-08-05 (1.552 kg known mass on the shin):
# effective joint Kt is 82-89% of datasheet. The model's force ranges already carry this factor.
KT_EFFICIENCY = 0.85


class JointMap:
    def __init__(self, entries=None, kt_joint=None, kt_efficiency=KT_EFFICIENCY):
        self.e = {}
        for name in MODEL_TO_MOTOR:
            d = dict((entries or {}).get(name, {}))
            self.e[name] = {"sign": float(d.get("sign", 1.0)),
                            "offset_deg": float(d.get("offset_deg", 0.0)),
                            "erpm_per_rad_s": d.get("erpm_per_rad_s"),
                            "verified": bool(d.get("verified", False)),
                            "verified_when": d.get("verified_when"),
                            "note": d.get("note", "")}
        self.kt_joint = (np.asarray(kt_joint, float) if kt_joint is not None
                         else np.array([4.655, 2.176, 2.176, 4.655, 2.176, 2.176]))
        self.kt_efficiency = float(kt_efficiency)
        # index of each model actuator inside a MOTOR_NAMES-ordered telemetry array
        self.model_from_motor = np.array([MOTOR_NAMES.index(n) for n in MODEL_TO_MOTOR], int)
        self.motor_from_model = np.array([MODEL_TO_MOTOR.index(n) for n in MOTOR_NAMES], int)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8-sig") as f:
            d = json.load(f)
        return cls(d.get("joints"), d.get("kt_joint"), d.get("kt_efficiency", KT_EFFICIENCY))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"joints": self.e, "kt_joint": self.kt_joint.tolist(),
                       "kt_efficiency": self.kt_efficiency}, f, indent=2)

    # ------------------------------------------------------------------ arrays, cached
    @property
    def sign(self):
        return np.array([self.e[n]["sign"] for n in MODEL_TO_MOTOR])

    @property
    def offset_rad(self):
        return np.radians([self.e[n]["offset_deg"] for n in MODEL_TO_MOTOR])

    # ------------------------------------------------------------------ conversions
    def to_model_rad(self, norm_deg_by_motor):
        """Telemetry in MOTOR_NAMES order (normalized degrees) -> model radians, actuator order."""
        v = np.asarray(norm_deg_by_motor, float)[self.model_from_motor]
        return np.radians(self.sign * v) + self.offset_rad

    def to_norm_deg(self, model_rad):
        """Model radians (actuator order) -> normalized degrees in MOTOR_NAMES order.

        The exact inverse of to_model_rad -- this is the direction the joint TARGETS travel, so an
        asymmetry between the two would put a constant bias on every command."""
        v = (np.degrees(np.asarray(model_rad, float) - self.offset_rad)) / self.sign
        return v[self.motor_from_model]

    def vel_to_model(self, erpm_by_motor, fallback_rad_s=None):
        """Reported ERPM -> model joint rad/s (actuator order), sign-corrected.

        Returns `fallback_rad_s` (already in actuator order -- typically the differentiated
        position) for any joint whose ERPM scale has not been fitted. Mixing a fitted joint with a
        differentiated one is fine and is better than guessing a scale: the observation the policy
        reads is per joint."""
        e = np.asarray(erpm_by_motor, float)[self.model_from_motor]
        out = np.zeros(len(MODEL_TO_MOTOR))
        for k, name in enumerate(MODEL_TO_MOTOR):
            s = self.e[name]["erpm_per_rad_s"]
            if s:
                out[k] = self.e[name]["sign"] * e[k] / float(s)
            elif fallback_rad_s is not None:
                out[k] = float(np.asarray(fallback_rad_s, float)[k])
            else:
                raise ValueError(
                    "{}: erpm_per_rad_s has never been measured and no differentiated-position "
                    "fallback was supplied. Nothing in this repo converts ERPM to joint speed -- "
                    "run fit_erpm_scale() on a hand-driven sweep.".format(name))
        return out

    def torque_to_model(self, amps_by_motor):
        """Reported current -> model joint torque, N*m, actuator order and model sign convention.

        Uses the MEASURED effective Kt (datasheet x kt_efficiency), not the datasheet value: the
        gearbox is ~85% efficient and the model's own force ranges already carry that factor, so
        using the datasheet here would inflate the torque channel of the observation by ~15%
        relative to what the policy saw in training."""
        a = np.asarray(amps_by_motor, float)[self.model_from_motor]
        return self.sign * a * self.kt_joint * self.kt_efficiency

    def amps_from_torque(self, tau_model):
        """Inverse of torque_to_model -- what a torque budget means in amps for the thermal model."""
        return np.abs(np.asarray(tau_model, float)) / np.maximum(
            self.kt_joint * self.kt_efficiency, 1e-9)

    # ------------------------------------------------------------------ gates
    def check_ready(self, need_velocity_scale=False):
        """(ok, message). The runner calls this BEFORE it is allowed to leave limp."""
        unver = [n for n in MODEL_TO_MOTOR if not self.e[n]["verified"]]
        if unver:
            return False, ("joint map not verified for {}. A sign error here drives a balance "
                           "correction the wrong way at 200 N*m/rad and looks exactly like a bad "
                           "policy. Run the verification procedure (robot limp, moved by hand) "
                           "before commanding anything.".format(", ".join(unver)))
        if need_velocity_scale:
            noscale = [n for n in MODEL_TO_MOTOR if not self.e[n]["erpm_per_rad_s"]]
            if noscale:
                return False, ("no measured ERPM scale for {} -- either fit it or run with the "
                               "differentiated-position fallback".format(", ".join(noscale)))
        return True, ""

    # ------------------------------------------------------------------ verification
    def check_stance(self, norm_deg_by_motor, expected_model_rad, tol_deg=8.0):
        """Static check: with the robot held in the model's stance pose, does the map agree?

        Returns (ok, per-joint report). This is the cheapest test that catches an offset error,
        and it is read-only -- the operator holds the pose, nothing is commanded. tol is generous
        on purpose: a human holding a 15 kg robot in a pose is worth a few degrees, and the test
        is looking for the 90/180-degree class of error, not for precision."""
        got = np.degrees(self.to_model_rad(norm_deg_by_motor))
        want = np.degrees(np.asarray(expected_model_rad, float))
        rep = []
        ok = True
        for k, name in enumerate(MODEL_TO_MOTOR):
            d = float(got[k] - want[k])
            good = abs(d) <= tol_deg
            ok = ok and good
            rep.append({"joint": name, "actuator": MODEL_ACTUATORS[k], "model_deg": float(got[k]),
                        "expected_deg": float(want[k]), "delta_deg": d, "ok": good})
        return ok, rep

    def check_direction(self, name, norm_before, norm_after, expect_model_increase):
        """Directional check: the operator moves ONE joint by hand in a named physical direction
        and we confirm the MODEL angle moves the way the model says it should.

        This is the half of the calibration a static pose cannot give you: an offset error and a
        sign error can produce the same reading at a single pose."""
        k = MODEL_TO_MOTOR.index(name)
        d = float(np.degrees(self.to_model_rad(norm_after) - self.to_model_rad(norm_before))[k])
        if abs(d) < 2.0:
            return False, "{} barely moved ({:+.1f} deg) -- move it further".format(name, d)
        good = (d > 0) == bool(expect_model_increase)
        return good, "{} moved {:+.1f} deg in model space (expected {})".format(
            name, d, "increase" if expect_model_increase else "decrease")

    def mark_verified(self, name, when=None, note=""):
        self.e[name]["verified"] = True
        self.e[name]["verified_when"] = when
        self.e[name]["note"] = note

    def invalidate(self, why=""):
        """Every drive re-randomises its raw origin on a power cycle, and the webui calibration
        that this map sits on top of is re-captured with it. When that happens the offsets here
        are stale too -- so a re-zero must clear these flags rather than leave a verified map
        describing a frame that no longer exists."""
        for n in MODEL_TO_MOTOR:
            self.e[n]["verified"] = False
            self.e[n]["note"] = why or "invalidated"


def fit_erpm_scale(erpm, norm_deg, dt, sign=1.0, min_speed_erpm=200.0):
    """Least-squares ERPM per joint rad/s from a hand-driven sweep of ONE joint (robot LIMP).

    Regresses reported ERPM against the differentiated normalized position. Samples below
    `min_speed_erpm` are dropped: near zero the ratio is dominated by the 0.1 degree position
    quantisation, and including them biases the slope toward zero.

    Returns (scale, r2, n_used). An r2 below ~0.95 means the sweep was too slow, too short, or
    the joint was not the only thing moving."""
    e = np.asarray(erpm, float)
    q = np.radians(np.asarray(norm_deg, float)) * float(sign)
    w = np.gradient(q, float(dt))
    m = np.abs(e) > float(min_speed_erpm)
    if m.sum() < 20:
        return None, 0.0, int(m.sum())
    x, y = w[m], e[m]
    slope = float((x @ y) / max(x @ x, 1e-12))
    resid = y - slope * x
    r2 = float(1.0 - (resid @ resid) / max(((y - y.mean()) @ (y - y.mean())), 1e-12))
    return slope, r2, int(m.sum())
