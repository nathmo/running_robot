/* DASH-01 web UI — vanilla JS, no external assets.
 * Data flow: /api/state @2 Hz (modes, calibration, files), /api/telemetry @10 Hz (samples,
 * linkage). All angles displayed/edited are NORMALIZED degrees (zero pose = 0). */
"use strict";

const MOTORS = ["right.abd", "right.cam", "right.thigh", "left.abd", "left.cam", "left.thigh"];
const ROLES = ["abd", "cam", "thigh"];
/* Fallback joint ranges, used only until /api/workspace has answered once. The daemon serves the
   real numbers (see _joint_limits in server.py) because two copies of a safety limit drift apart:
   this literal said the thigh stopped at 62 deg while the daemon had already widened itself to the
   demonstrated workspace, and the workspace editor clipped strokes at whichever number it held. */
const HARD_FALLBACK = { abd: 48, cam: 88, thigh: 62 };

/* kind: "hard" = the never-exceed refusal, as far as a region may be drawn;
         "nominal" = the CAD guess widened by what has been demonstrated, what the UI spans. */
function jointLimit(side, role, kind) {
  const L = S.ws && S.ws.limits && S.ws.limits[side] && S.ws.limits[side][role];
  if (L && L[kind]) return L[kind];
  return [-HARD_FALLBACK[role], HARD_FALLBACK[role]];
}
const COLORS = { pos: "#4da3ff", target: "#f2f2f2", cur: "#e0a020", temp: "#e04545", good: "#2c9e3f",
  region: "rgba(44,158,63,0.55)", samples: "rgba(160,170,185,0.4)", gait: "#ff35c8",
  stroke: "#ffd23f", zero: "#ffffff" };

const S = {
  state: null, token: null, seq: 0,
  wsLeg: "right", trajLeg: "right",
  ws: null,                      // /api/workspace payload
  traj: null, trajName: null,    // shown trajectory
  latest: {},                    // motor -> latest telemetry values
  chartSpan: 120,                // seconds shown per telemetry chart (buffers keep CHART_KEEP_S)
  // Last KNOWN value of fields that only some state payloads carry. /api/state merges the hot and
  // cold halves, but api() calls applyState with the daemon snapshot alone, which has no
  // calibration and no workspace. Reading those directly made the "calibrated" badge flip to NOT
  // CALIBRATED on every API call; the lamps below would have inherited the same flicker. Only ever
  // LEARN a field that is present.
  cal: {}, ws: {}, bypassBanner: false,
  mockTimers: {},
  preview: { on: false, t0: 0 },  // client-side both-legs gait preview animation
  sineDefaults: {},               // motor -> {a,b,center,...} 70%-of-safe-range sine presets
  sineDefFetched: false,          // presets pulled once after calibration completes
  wsTrail: [],                    // live (cam,thigh) trail accumulated during a workspace sweep
  wsAbdSweep: [Infinity, -Infinity],  // live min/max abduction swept during a sweep
};

/* ================================================================ tiny helpers */
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (+v).toFixed(d);

/* "right ✓ / left —" pills that spell out which legs a workspace or gait actually contains */
function legBadges(el, hasRight, hasLeft) {
  if (!el) return;
  el.innerHTML =
    `<span class="leg-badge ${hasRight ? "has" : ""}">right ${hasRight ? "✓" : "—"}</span>` +
    `<span class="leg-badge ${hasLeft ? "has" : ""}">left ${hasLeft ? "✓" : "—"}</span>`;
}

async function api(path, opts = {}) {
  const o = { headers: {}, ...opts };
  if (o.json !== undefined) {
    o.method = o.method || "POST";
    o.headers["Content-Type"] = "application/json";
    if (S.token) o.json.token = S.token;
    o.body = JSON.stringify(o.json);
    delete o.json;
  }
  let r;
  try { r = await fetch(path, o); } catch (e) { setBanner("connection lost: " + e, "error"); throw e; }
  let d = null;
  try { d = await r.json(); } catch (_) { /* file downloads etc. */ }
  if (d && d.token) S.token = d.token;
  if (d && d.ok === false) { setBanner(d.error, "error", 6000); throw new Error(d.error); }
  if (d && d.state) applyState(d.state);
  return d;
}

let bannerTimer = null;
function setBanner(msg, cls = "", ms = 0) {
  const b = $("banner");
  if (!msg) { b.classList.add("hidden"); return; }
  b.textContent = msg;
  b.className = "banner " + cls;
  if (bannerTimer) clearTimeout(bannerTimer);
  if (ms) bannerTimer = setTimeout(() => b.classList.add("hidden"), ms);
}

/* ================================================================ state polling */
async function pollState() {
  try {
    const st = await (await fetch("/api/state")).json();
    applyState(st);
  } catch (e) { $("daemon-dot").classList.remove("alive"); }
}

function applyState(st) {
  if (!st || !st.mode) return;
  S.state = st;
  const mode = st.mode;
  const mb = $("mode-badge");
  mb.textContent = mode;
  mb.className = "badge mode-" + mode;
  if (st.calibration) S.cal = st.calibration;
  if (st.workspace) S.ws = st.workspace;
  const cal = S.cal;
  const cb = $("calib-badge");
  cb.textContent = cal.stage === "complete" ? "calibrated" : "NOT CALIBRATED";
  cb.className = "badge " + (cal.stage === "complete" ? "cal-ok" : "cal-no");
  updateLamps(st);
  updateGuards(st);
  $("daemon-dot").classList.toggle("alive", !!st.daemon_alive && st.daemon_thread_alive !== false);
  $("loop-info").textContent = st.loop ? `${st.loop.hz | 0} Hz, slip ${st.loop.slip}` : "";

  if (st.loop_error) setBanner("DAEMON CRASHED (motors limp): " + st.loop_error.split("\n").pop(), "error");
  else if (st.estop && st.estop.latched) setBanner("E-STOPPED: " + (st.estop.reason || "") +
    " — clear with the E-STOP button", "error");
  else if (cal.restored_from_disk && cal.stage === "complete")
    setBanner("Calibration restored from disk — valid ONLY if motors were NOT power-cycled since. Re-zero if unsure.", "warn");

  const rj = $("reject-banner");
  if (st.last_reject) { rj.textContent = "⛔ " + st.last_reject; rj.classList.remove("hidden"); }
  else rj.classList.add("hidden");

  updateWizard(st);
  updateMotorCards(st);
  updateRecordUI(st);
  updatePlaybackUI(st);
  updateManualStatus(st);
  updateFileLists(st);
  // the visibility toggle lives on the ROW, not the panel: a hidden panel inside a visible
  // (empty) row would still cost the page an extra flex gap
  $("row-mock").classList.toggle("hidden", !st.mock);
  $("btn-estop").textContent = (st.estop && st.estop.latched) ? "CLEAR E-STOP" : "E-STOP";
  if (cal.stage === "complete" && !S.sineDefFetched) { S.sineDefFetched = true; fetchSineDefaults(); }
  if (window.onSysidState) window.onSysidState(st);      // system-ID panels (sysid.js)
  if (window.onThermalState) window.onThermalState(st); // thermal calibration (thermal.js)
  if (window.onPolicyState) window.onPolicyState(st);   // policy runs (policy.js)
}

/** Has the slow guided move (Home / ⌖ Centre) reached what it was sent to? The move stays armed
 *  after arrival — it keeps holding the pose — so "arrived" has to be measured, and against the
 *  PUBLISHED targets: ⌖ Centre drives to the max-room pose, which is not the zero pose.
 *  Shared with sysid.js, which reports the same move in the system-ID panel. */
function atManualTarget(man) {
  const tgt = (man && man.targets) || {};
  return MOTORS.every((n) => S.latest[n] && S.latest[n].pos_norm !== null &&
                             Math.abs(S.latest[n].pos_norm - (tgt[n] || 0)) < 1.0);
}

/* homing banner + keep the override checkbox in sync with the daemon */
function updateManualStatus(st) {
  const man = st.manual || {};
  if (man.homing) {
    const centring = man.homing_kind === "center";
    const arrived = atManualTarget(man);
    $("home-status").textContent = man.homing_kind === "balance_hold"
      ? "⚖ balance stopped — holding its last pose"
      : centring
      ? (arrived ? "⌖ centred ✓ (most room around this pose)" : "⌖ centring… (slow)")
      : (arrived ? "🏠 at home ✓ (holding the standing pose)" : "🏠 homing… (slow)");
  } else {
    $("home-status").textContent = "";
  }
  updateBalanceStatus(man.balance);
  const chk = $("chk-override");
  if (document.activeElement !== chk && man.override !== undefined) chk.checked = !!man.override;
}

/* ================================================================ e-stop / header */
$("btn-estop").onclick = async () => {
  const latched = S.state && S.state.estop && S.state.estop.latched;
  await api(latched ? "/api/estop/clear" : "/api/estop", { method: "POST" });
};
$("btn-limp").onclick = () => api("/api/mode", { json: { mode: "limp" } });
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
    e.preventDefault();
    api("/api/estop", { method: "POST" });
  }
});

/* ================================================================ calibration wizard */
const INSTR = {
  "right.abd": "Abduct the RIGHT leg so the FOOT LIFTS OFF THE GROUND.",
  "right.cam": "Backdrive the RIGHT cam DOWN.",
  "right.thigh": "Swing the RIGHT thigh FORWARD.",
  "left.abd": "Abduct the LEFT leg so the FOOT LIFTS OFF THE GROUND.",
  "left.cam": "Backdrive the LEFT cam DOWN.",
  "left.thigh": "Swing the LEFT thigh FORWARD.",
};

function updateWizard(st) {
  const cal = st.calibration || {};
  const complete = cal.stage === "complete";
  $("wiz-steps").classList.toggle("hidden", complete);
  $("wiz-summary").classList.toggle("hidden", !complete);
  $("panel-calib").classList.toggle("attention", !complete);
  // gate every actionable panel until calibrated (telemetry + mock stay usable — the wizard
  // itself needs live values and mock dragging)
  document.querySelectorAll("#main .panel").forEach((p) => {
    // panel-limbs and panel-policy are pure inspection here (mass entry / bundle metadata + a
    // mock-bus rehearsal that energises nothing) — usable before calibration
    if (!["panel-calib", "panel-telemetry", "panel-mock", "panel-files", "panel-limbs",
          "panel-policy"].includes(p.id))
      p.classList.toggle("locked", !complete);
  });
  if (complete) {
    $("calib-summary-text").textContent = `✓ calibrated (${cal.created || "unknown time"})` +
      (cal.restored_from_disk ? " — RESTORED FROM DISK: only valid if motors were not power-cycled since" : "");
    return;
  }
  const step2 = cal.stage === "zero_set";
  $("wiz-step1").classList.toggle("hidden", step2);
  $("wiz-step2").classList.toggle("hidden", !step2);

  if (!step2) {
    const live = $("wiz-live");
    live.innerHTML = MOTORS.map((n) => {
      const m = (st.motors || {})[n] || {};
      return `<div>${n}<br><b>${fmt(m.pos_raw)}°</b> raw ${m.alive ? "" : " ⚠ silent"}</div>`;
    }).join("");
  } else {
    const cards = $("wiz-cards");
    if (!cards.dataset.built) {
      cards.dataset.built = "1";
      cards.innerHTML = MOTORS.map((n) => `
        <div class="wiz-card" id="wc-${n.replace(".", "-")}">
          <b>${n}</b> <span class="sign"></span>
          <div class="val">—</div>
          <div class="instr">${INSTR[n]}</div>
          <div class="row">
            <button class="btn small" data-zero1="${n}" title="Re-capture THIS joint's zero from
where it is right now — pose just this joint, leave the other five alone. Its direction must be
re-confirmed afterwards (the sign is kept, the ✓ is cleared).">📍 Zero</button>
            <button class="btn small" data-flip="${n}">↔ Flip</button>
            <button class="btn small primary" data-confirm="${n}">✓ Confirm</button>
          </div>
        </div>`).join("");
      cards.querySelectorAll("[data-zero1]").forEach((b) => b.onclick = () =>
        api("/api/calibration/zero_one", { json: { motor: b.dataset.zero1 } }));
      cards.querySelectorAll("[data-flip]").forEach((b) => b.onclick = async () => {
        const n = b.dataset.flip;
        const cur = S.state.calibration.motors[n].sign;
        await api("/api/calibration/sign", { json: { motor: n, sign: -cur } });
      });
      cards.querySelectorAll("[data-confirm]").forEach((b) => b.onclick = () =>
        api("/api/calibration/confirm", { json: { motor: b.dataset.confirm } }));
    }
    let allOk = true;
    for (const n of MOTORS) {
      const card = $("wc-" + n.replace(".", "-"));
      const mc = cal.motors[n] || {};
      const tv = (st.motors || {})[n] || {};
      card.classList.toggle("confirmed", !!mc.confirmed);
      card.querySelector(".val").textContent = fmt(tv.pos_norm) + "°";
      card.querySelector(".sign").textContent = `sign ${mc.sign > 0 ? "+1" : "−1"}` +
        (mc.confirmed ? " ✓" : "");
      allOk = allOk && mc.confirmed;
    }
    $("btn-wiz-done").disabled = !allOk;
  }
}
$("btn-set-zero").onclick = () => api("/api/calibration/zero", { method: "POST" });
$("btn-wiz-back").onclick = () => api("/api/calibration/reset", { method: "POST" });
$("btn-wiz-done").onclick = () => api("/api/calibration/complete", { method: "POST" });
$("btn-recal-zero").onclick = async () => {
  if (!confirm("Re-zero: pose the robot at the URDF zero pose FIRST, then OK. " +
               "Directions must be re-confirmed afterwards (they can invert across power cycles).")) return;
  await api("/api/calibration/zero", { method: "POST" });   // -> stage zero_set, wizard reopens
};
$("btn-recal-reset").onclick = () => {
  if (confirm("Reset calibration entirely? All motion stays locked until the wizard is redone."))
    api("/api/calibration/reset", { method: "POST" });
};

/* ================================================================ strip charts */
/* The longest window the buffers RETAIN, independent of what is currently displayed. Keeping the
 * full history means widening the time window shows history that already happened, instead of
 * starting an empty chart and making you wait for it. At 10 Hz x 6 motors x 3 charts this is a few
 * hundred kB, which is nothing next to being able to look backwards. */
const CHART_KEEP_S = 600;

class StripChart {
  /** color2, when given, draws an optional second series (the commanded target) under the first. */
  constructor(canvas, color, span = 60, color2 = null) {
    this.cv = canvas; this.color = color; this.color2 = color2; this.span = span;
    this.t = []; this.v = []; this.v2 = [];
    // OPT-IN, and it must stay opt-in. _fit() sizes the backing store from the element's measured
    // box; on a canvas with no CSS width the width ATTRIBUTE is what drives layout, so writing it
    // changes the measurement that produced it and the canvas grows without bound. That is what
    // happened to #meas-chart on 2026-08-29 — it has no CSS width, so it doubled every frame and,
    // sitting in a 1fr grid, dragged Gait playback / Dynamic ID / Limbs & inertia / Manage files
    // out to infinity with it. Only enable this on a canvas whose width comes from CSS.
    this.autoFit = false;
  }
  /** v2 is optional and may be null on any sample — nothing is commanded while limp, and the line
   *  must BREAK there rather than hold the last target flat, which would read as a real command. */
  push(t, v, v2) {
    if (v === null || v === undefined) return;
    this.t.push(t); this.v.push(v);
    this.v2.push(v2 === null || v2 === undefined ? NaN : v2);
    const cut = t - CHART_KEEP_S;
    while (this.t.length && this.t[0] < cut) { this.t.shift(); this.v.shift(); this.v2.shift(); }
  }
  /** Match the backing store to the CSS box, in device pixels, so a full-width chart is crisp
   *  rather than a 220 px bitmap stretched across 700 px. */
  _fit() {
    if (!this.autoFit) { this.dpr = 1; return; }
    const c = this.cv, r = c.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    const w = Math.max(80, Math.round(r.width * this.dpr));
    const h = Math.max(24, Math.round(r.height * this.dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  }
  /** Time gridlines with real labels. Without them a strip chart shows you a shape but not how
   *  long anything took, which is the question you actually have when watching a gait. */
  _timeAxis(g, w, h) {
    const k = this.dpr || 1;
    // ~5 divisions, snapped to a round number of seconds
    const raw = this.span / 5;
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300].find((x) => x >= raw) || 300;
    g.save();
    g.font = `${Math.round(9 * k)}px monospace`;
    g.textBaseline = "bottom";
    for (let a = 0; a <= this.span + 1e-6; a += step) {
      const x = w - (a / this.span) * w;
      g.strokeStyle = a === 0 ? "rgba(139,151,168,.35)" : "rgba(139,151,168,.14)";
      g.lineWidth = k;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
      if (!this.showTimeLabels) continue;
      g.fillStyle = "rgba(139,151,168,.75)";
      g.textAlign = a === 0 ? "right" : (x < 24 * k ? "left" : "center");
      g.fillText(a === 0 ? "now" : `-${a}s`, a === 0 ? w - 2 * k : x, h - 2 * k);
    }
    g.restore();
  }
  _stroke(g, vals, now, lo, hi, w, h) {
    let pen = false;
    g.beginPath();
    for (let i = 0; i < this.t.length; i++) {
      if (!Number.isFinite(vals[i])) { pen = false; continue; }
      const x = ((this.t[i] - now + this.span) / this.span) * w;
      const y = h - ((vals[i] - lo) / (hi - lo)) * h;
      pen ? g.lineTo(x, y) : g.moveTo(x, y);
      pen = true;
    }
    g.stroke();
  }
  draw(now, refLine = null) {
    const c = this.cv, g = c.getContext("2d");
    this._fit();
    const k = this.dpr || 1;
    const w = c.width, h = c.height;
    g.clearRect(0, 0, w, h);
    this._timeAxis(g, w, h);
    if (this.v.length < 2) return;
    // Scale over BOTH series: if the target is clipped out of view you cannot see the tracking
    // error, which is the entire reason for drawing it.
    const finite2 = this.color2 ? this.v2.filter(Number.isFinite) : [];
    let lo = Math.min(...this.v, ...finite2), hi = Math.max(...this.v, ...finite2);
    if (refLine !== null) { lo = Math.min(lo, refLine - 5); hi = Math.max(hi, refLine + 5); }
    if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
    if (refLine !== null && refLine <= hi && refLine >= lo) {
      const y = h - ((refLine - lo) / (hi - lo)) * h;
      g.strokeStyle = "rgba(224,69,69,.5)"; g.setLineDash([3 * k, 3 * k]);
      g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); g.setLineDash([]);
    }
    if (this.color2 && finite2.length) {
      g.strokeStyle = this.color2; g.lineWidth = k; g.setLineDash([4 * k, 3 * k]);
      this._stroke(g, this.v2, now, lo, hi, w, h);
      g.setLineDash([]);
    }
    g.strokeStyle = this.color; g.lineWidth = 1.2 * k;
    this._stroke(g, this.v, now, lo, hi, w, h);
    g.fillStyle = "rgba(139,151,168,.9)";
    g.font = `${Math.round(9 * k)}px monospace`;
    g.textAlign = "left"; g.textBaseline = "alphabetic";
    g.fillText(hi.toFixed(1), 2 * k, 9 * k);
    // lift the range label clear of the time axis on whichever chart carries the labels
    g.fillText(lo.toFixed(1), 2 * k, h - (this.showTimeLabels ? 12 : 2) * k);
  }
}


/* Two things the operator has to know before commanding anything, and which are otherwise buried
 * in the calibration wizard and the workspace panel.
 *
 * ZEROED is not the same question as "is there a calibration". Every drive re-randomises its raw
 * encoder origin on a power cycle, so a calibration restored from disk describes a frame that no
 * longer exists — the daemon's pre-move guard refuses motion until it is re-captured. Amber is
 * exactly that state: calibrated, but not in THIS session, so nothing will move yet.
 *
 * WORKSPACE is the recorded feasible-configuration set the daemon bounds MANUAL, PLAYBACK and the
 * identify wiggle against. Without it _safe_room has nothing to say and those guards fall back to
 * the joints' own hard limits. */
function updateLamps(st) {
  const cal = S.cal, ws = S.ws;
  const guard = (st.premove || {}).refused || "";
  const zl = $("lamp-zero");
  if (cal.stage !== "complete") {
    zl.className = "lamp off";
    zl.title = "No calibration. Run the zero/direction wizard — nothing will move until you do.";
  } else if (cal.restored_from_disk) {
    zl.className = "lamp warn";
    zl.title = "Calibration restored from disk, not re-captured this session. The drives "
      + "re-randomise their raw origin on every power cycle, so the pre-move guard will refuse "
      + "motion until you set zero again.";
  } else if (guard) {
    zl.className = "lamp warn";
    zl.title = "Zeroed this session, but the pre-move guard is refusing: " + guard;
  } else {
    zl.className = "lamp ok";
    zl.title = `Zeroed this session (epoch ${cal.zero_epoch}). The pre-move guard is satisfied.`;
  }

  const wl = $("lamp-ws");
  const legs = (ws.legs || []).length;
  if (!ws.source) {
    wl.className = "lamp off";
    wl.title = "No safe workspace loaded. MANUAL, playback and the identify wiggle fall back to "
      + "each joint's own hard limits — the self-collision check has nothing to work from.";
  } else if (legs < 2) {
    wl.className = "lamp warn";
    wl.title = `${ws.source}: only ${legs} leg recorded. The missing side is unbounded by the `
      + "workspace check.";
  } else {
    wl.className = "lamp ok";
    wl.title = `${ws.source} — both legs recorded.`;
  }
}


/* Four software limits the operator can switch off. Ticked = enforced; unticking DISABLES the
 * limit and asks first, naming what stops being checked. The daemon is the authority — these
 * checkboxes only ever mirror st.bypass, so a bypass set from another tab shows up here too.
 *
 * What none of them touch: the joints' own hard limits, the drive's firmware current limit, the
 * E-STOP, the fall kill, drive errors, over-temperature, telemetry staleness. */
const GUARD_WARN = {
  workspace: "Disable the WORKSPACE limit?\n\nPoses will no longer be checked against the "
    + "recorded feasible set, so the legs may be commanded into each other or into the ground.\n\n"
    + "Each joint's own hard limit still applies — this cannot drive a joint into a stop.",
  speed: "Disable the SPEED limit?\n\nThe playback speed governor and the runaway cut-out both "
    + "stop applying. A joint that starts accelerating will not be caught by software.",
  torque: "Disable the TORQUE limit?\n\nThe playback current cap rises from its configured value "
    + "to the 40 A software ceiling. The drive's own firmware phase limit is still the real "
    + "ceiling and is not affected.",
  tracking: "Disable the TRACKING-ERROR limit?\n\nThe daemon will no longer trip when a joint "
    + "falls behind its commanded position. That trip is what catches a stale zero, a jammed "
    + "joint, and a mechanism that has stopped following — before it is obvious.",
};

function wireGuards() {
  document.querySelectorAll(".gk").forEach((el) => {
    el.onchange = async () => {
      const name = el.dataset.bypass;
      const disabling = !el.checked;            // unticking = turning the SAFETY off
      if (disabling && !confirm(GUARD_WARN[name] + "\n\nThis is recorded in the flight recorder "
          + "and stays off until the daemon restarts.")) {
        el.checked = true;                      // refused: put the guard back
        return;
      }
      try {
        await api("/api/bypass", { json: { name, on: disabling, acknowledged: true } });
      } catch (_) { el.checked = !disabling; }  // server refused: mirror reality
    };
  });
}

function updateGuards(st) {
  const by = st.bypass;
  if (!by) return;                              // partial state: never guess
  let any = false;
  document.querySelectorAll(".gk").forEach((el) => {
    const off = !!by[el.dataset.bypass];
    if (document.activeElement !== el) el.checked = !off;
    el.parentElement.classList.toggle("off", off);
    any = any || off;
  });
  $("guards").classList.toggle("breached", any);
  const names = Object.keys(by).filter((k) => by[k]);
  if (names.length) {
    setBanner("SAFETY BYPASSED: " + names.join(", ") + " — these limits are NOT being enforced. "
      + "They re-arm when the daemon restarts.", "error");
    S.bypassBanner = true;
  } else if (S.bypassBanner) {
    S.bypassBanner = false;
    setBanner("");
  }
}

/* ================================================================ telemetry cards */
const charts = {};
function buildMotorCards() {
  $("motor-cards").innerHTML = MOTORS.map((n) => `
    <div class="motor-card" id="mc-${n.replace(".", "-")}">
      <div class="mc-title">${n}</div>
      <div class="mc-vals">
        <span>norm <b class="v-norm">—</b>°</span><span>raw <span class="v-raw">—</span>°</span>
        <span><span class="v-cur">—</span> A</span><span><span class="v-temp">—</span> °C</span>
        <span class="mc-err v-err"></span>
      </div>
      <div class="mc-plot"><span class="mc-ylab">pos °</span><canvas class="c-pos"></canvas></div>
      <div class="mc-plot"><span class="mc-ylab">current A</span><canvas class="c-cur"></canvas></div>
      <div class="mc-plot"><span class="mc-ylab">temp °C</span><canvas class="c-temp"></canvas></div>
    </div>`).join("");
  for (const n of MOTORS) {
    const card = $("mc-" + n.replace(".", "-"));
    charts[n] = {
      pos: new StripChart(card.querySelector(".c-pos"), COLORS.pos, S.chartSpan, COLORS.target),
      cur: new StripChart(card.querySelector(".c-cur"), COLORS.cur, S.chartSpan),
      temp: new StripChart(card.querySelector(".c-temp"), COLORS.temp, S.chartSpan),
    };
    // these three are `width: 100%` in the stylesheet, so their layout width comes from CSS and
    // sizing the backing store cannot feed back into it
    for (const k of ["pos", "cur", "temp"]) charts[n][k].autoFit = true;
    // only the bottom chart of each card carries the labels; the gridlines line up across all
    // three, so one set of numbers reads for the whole column
    charts[n].temp.showTimeLabels = true;
  }
  const sel = $("tel-span");
  if (sel) {
    sel.value = String(S.chartSpan);
    sel.onchange = () => {
      S.chartSpan = +sel.value;
      for (const n of MOTORS) {
        for (const k of ["pos", "cur", "temp"]) charts[n][k].span = S.chartSpan;
      }
      drawCharts();
    };
  }
}

function updateMotorCards(st) {
  for (const n of MOTORS) {
    const card = $("mc-" + n.replace(".", "-"));
    if (!card) continue;
    const m = (st.motors || {})[n] || {};
    card.classList.toggle("dead", !m.alive);
    card.querySelector(".v-norm").textContent = fmt(m.pos_norm);
    card.querySelector(".v-raw").textContent = fmt(m.pos_raw);
    card.querySelector(".v-cur").textContent = fmt(m.cur, 2);
    card.querySelector(".v-temp").textContent = m.temp ?? "—";
    card.querySelector(".v-err").textContent = m.err ? ("ERR " + m.err) : (m.alive ? "" : "silent");
  }
}

/* ================================================================ telemetry polling */
async function pollTelemetry() {
  try {
    const d = await (await fetch(`/api/telemetry?since=${S.seq}`)).json();
    S.seq = d.seq;
    const n = d.t.length;
    for (const name of MOTORS) {
      const m = d.motors[name];
      for (let i = 0; i < n; i++) {
        charts[name].pos.push(d.t[i], m.pos_norm[i], m.cmd_norm ? m.cmd_norm[i] : null);
        charts[name].cur.push(d.t[i], m.cur[i]);
        charts[name].temp.push(d.t[i], m.temp[i]);
      }
      if (n) S.latest[name] = { pos_norm: m.pos_norm[n - 1], pos_raw: m.pos_raw[n - 1],
        cur: m.cur[n - 1], temp: m.temp[n - 1] };
    }
    if (n) S.lastT = d.t[n - 1];
    accumulateWsTrail(d);
  } catch (e) { /* banner handled by state poll */ }
}

function drawCharts() {
  const now = S.lastT || 0;
  for (const n of MOTORS) {
    if (!document.hidden) {
      charts[n].pos.draw(now);
      charts[n].cur.draw(now);
      charts[n].temp.draw(now, 80);
    }
  }
}

/* ================================================================ manual control */
/* Each slider is a live gauge you can grab: when idle it tracks the motor's real position; while
 * you drag it (or type in the number box) it commands a target. The number box mirrors the slider
 * — live when idle, an exact setpoint when you type one. Sine start/stop preset to 70% of the safe
 * range around the current pose (fetched from the daemon). manRow[n] tracks per-row edit state. */
const manualDesired = {}; let manualDirty = false;
const manRow = {};                 // n -> {editing, lastInteract, sineEdited}
const GRACE_MS = 350;              // after a drag/edit, keep the value before resuming live-follow

function buildManualRows() {
  $("manual-rows").innerHTML = MOTORS.map((n) => {
    const role = n.split(".")[1];
    return `
    <div class="man-row" id="man-${n.replace(".", "-")}">
      <span class="mr-line mr-line1">
        <span class="mr-name">${n}</span>
        <input type="range" class="mr-slider" min="${-HARD_FALLBACK[role]}" max="${HARD_FALLBACK[role]}" step="0.5" value="0">
      </span>
      <span class="mr-line mr-line2">
        <input type="number" class="num mr-num" step="0.5" value="0">
        <span class="mr-target"></span>
      </span>
      <span class="mr-line man-sine">
        <label><input type="checkbox" class="sn-en">sine</label>
        <input type="number" class="num sn-a" title="start angle ° (preset to 70% of the safe range)">↔<input type="number" class="num sn-b" title="stop angle ° (preset to 70% of the safe range)">
        <input type="number" class="num sn-f" value="0.3" step="0.05" min="0.02" max="3" title="Hz">Hz
        <button class="btn small sn-auto" title="reset start/stop to 70% of the safe range around the current position">↺</button>
      </span>
    </div>`;
  }).join("");
  for (const n of MOTORS) {
    manRow[n] = { editing: false, lastInteract: 0, sineEdited: false };
    const row = $("man-" + n.replace(".", "-"));
    const slider = row.querySelector(".mr-slider"), num = row.querySelector(".mr-num");
    const command = (v) => { manualDesired[n] = +v; manualDirty = true; manRow[n].lastInteract = performance.now(); };
    const release = () => { manRow[n].editing = false; manRow[n].lastInteract = performance.now(); };
    slider.addEventListener("pointerdown", () => { manRow[n].editing = true; });
    slider.addEventListener("pointerup", release);
    slider.addEventListener("pointercancel", () => { manRow[n].editing = false; });
    slider.oninput = () => { num.value = (+slider.value).toFixed(1); command(slider.value); updateManualTargetHint(n); };
    num.onfocus = () => { manRow[n].editing = true; };
    num.onblur = release;
    num.onchange = () => { slider.value = num.value; command(num.value); updateManualTargetHint(n); };

    // ---- sine ----
    const sineSend = () => api("/api/sine", { json: {
      actuator: n, enabled: row.querySelector(".sn-en").checked,
      a_deg: +row.querySelector(".sn-a").value, b_deg: +row.querySelector(".sn-b").value,
      freq_hz: +row.querySelector(".sn-f").value } });
    row.querySelector(".sn-en").onchange = async () => {
      if (row.querySelector(".sn-en").checked && !manRow[n].sineEdited) {
        if (!S.sineDefaults[n]) await fetchSineDefaults();
        applySineDefault(n);
        if (row.querySelector(".sn-a").value === "") row.querySelector(".sn-a").value = -5;
        if (row.querySelector(".sn-b").value === "") row.querySelector(".sn-b").value = 5;
      }
      sineSend();
    };
    row.querySelectorAll(".sn-a,.sn-b").forEach((i) =>
      i.addEventListener("input", () => { manRow[n].sineEdited = true; }));
    row.querySelectorAll(".sn-a,.sn-b,.sn-f").forEach((i) => i.onchange = () => {
      if (row.querySelector(".sn-en").checked) sineSend();
    });
    row.querySelector(".sn-auto").onclick = async () => {
      manRow[n].sineEdited = false;
      await fetchSineDefaults();
      applySineDefault(n, true);
      if (row.querySelector(".sn-en").checked) sineSend();
    };
  }
}

/* slider/number follow the live motor position whenever the row is idle (not being dragged/typed,
 * and past the post-release grace window). Purely display — never sets a target. */
function updateManualLive() {
  const now = performance.now();
  for (const n of MOTORS) {
    const st = manRow[n];
    if (!st) continue;
    const row = $("man-" + n.replace(".", "-"));
    if (!row) continue;
    const live = S.latest[n] ? S.latest[n].pos_norm : null;
    const idle = !st.editing && (now - st.lastInteract > GRACE_MS);
    if (idle && live !== null && live !== undefined) {
      row.querySelector(".mr-slider").value = live;
      const num = row.querySelector(".mr-num");
      if (document.activeElement !== num) num.value = (+live).toFixed(1);
    }
    updateManualTargetHint(n);
  }
}

/* small "⇒ target°" hint next to the number while the motor is still slewing to a commanded pose */
function updateManualTargetHint(n) {
  const row = $("man-" + n.replace(".", "-"));
  if (!row) return;
  const el = row.querySelector(".mr-target");
  const live = S.latest[n] ? S.latest[n].pos_norm : null;
  const tgt = manualDesired[n];
  const manualMode = S.state && S.state.mode === "MANUAL";
  el.textContent = (manualMode && tgt !== undefined && live !== null &&
                    Math.abs(tgt - live) > 1.5) ? `⇒ ${(+tgt).toFixed(1)}°` : "";
}

setInterval(() => {         // 20 Hz slider flush (one final value lands after release too)
  if (!manualDirty) return;
  manualDirty = false;
  api("/api/manual", { json: { targets: { ...manualDesired },
    override: $("chk-override").checked, slew_dps: +$("inp-slew").value } }).catch(() => {});
}, 50);

/* ---- sine 70%-of-safe-range presets ---- */
let sineDefPending = null;
async function fetchSineDefaults() {
  const cal = S.state && S.state.calibration;
  if (!(cal && cal.stage === "complete")) return;
  if (sineDefPending) return sineDefPending;
  sineDefPending = api("/api/manual/sine_defaults", { json: {} })
    .then((d) => {
      if (d && d.defaults) {
        S.sineDefaults = d.defaults;
        for (const n of MOTORS) applySineDefault(n);      // fill every un-edited row
      }
    }).catch(() => {}).finally(() => { sineDefPending = null; });
  return sineDefPending;
}
function applySineDefault(n, force = false) {
  const d = S.sineDefaults[n];
  if (!d || (manRow[n].sineEdited && !force)) return;
  const row = $("man-" + n.replace(".", "-"));
  if (!row) return;
  row.querySelector(".sn-a").value = d.a;
  row.querySelector(".sn-b").value = d.b;
}

$("btn-hold").onclick = async () => {
  // enter manual at the current pose: seed sliders from live positions
  for (const n of MOTORS) {
    const v = S.latest[n] ? S.latest[n].pos_norm : 0;
    const row = $("man-" + n.replace(".", "-"));
    row.querySelector(".mr-slider").value = v;
    row.querySelector(".mr-num").value = (+v).toFixed(1);
    manualDesired[n] = v;
    manRow[n].lastInteract = 0;
  }
  await api("/api/manual", { json: { targets: { ...manualDesired } } });
  fetchSineDefaults();
};
$("btn-home").onclick = async () => {
  await api("/api/manual/home", { json: { slew_dps: +$("inp-home-slew").value } });
  setBanner("homing to the standing pose (slow)…", "", 4000);
};
$("btn-center").onclick = async () => {
  const d = await api("/api/manual/center", { json: { slew_dps: +$("inp-home-slew").value } });
  if (window.onCenterResult) window.onCenterResult(d);       // system-ID panel trims its amplitudes
  setBanner("centring both legs on the safest pose (slow)…", "", 4000);
};
$("btn-release").onclick = () => api("/api/manual/release", { method: "POST" });

/* ⚖ Balance (daemon.balance_start / balance.py): start/stop + the operator trims */
const BAL = { active: false };
function updateBalanceStatus(b) {
  if (!b) return;
  BAL.active = !!b.active;
  const btn = $("btn-balance");
  btn.textContent = BAL.active ? "⚖ Balance: STOP (hold)" : "⚖ Balance: start";
  btn.classList.toggle("active-rec", BAL.active);
  const o = b.out;
  const sg = (v, d = 1) => (v >= 0 ? "+" : "") + (+v).toFixed(d);
  $("bal-status").textContent = BAL.active && o
    ? `⚖ tilt p ${sg(o.pitch)}° r ${sg(o.roll)}° → posture ${sg(o.pitch_corr)}°, CoM x ${sg(o.com_x)} y ${sg(o.com_y)} mm` +
      (o.saturated ? "  ⚠ at its limit" : "")
    : "";
  const t = b.trim || {};
  for (const [id, key] of [["bal-comx", "com_x_mm"], ["bal-comy", "com_y_mm"], ["bal-pitch", "pitch_deg"]]) {
    const el = $(id);
    // A value the operator just set is DIRTY until the daemon has answered the POST that carries
    // it. Until then a state poll must not write the daemon's copy back over it: the poll answers
    // every 500 ms and one is nearly always in flight with the OLD trim, and the ◀ ▶ buttons take
    // the focus, so "not the active element" protected nothing -- the click was snapped back to
    // the old value inside the 120 ms send debounce and the old value is what got sent.
    if (el.dataset.dirty && el.dataset.want !== undefined &&
        (String(t[key]) === el.dataset.want || Date.now() - +el.dataset.wantT > 2000)) {
      delete el.dataset.dirty; delete el.dataset.want; delete el.dataset.wantT;   // echoed (or given up)
    }
    if (document.activeElement !== el && !el.dataset.dirty && t[key] !== undefined) el.value = t[key];
  }
  $("bal-comx-val").textContent = sg($("bal-comx").value, 0);
  $("bal-comy-val").textContent = sg($("bal-comy").value, 0);
  // where the trim lands: it re-poses the robot only while it holds the standing pose (Home
  // arrived / Balance stopped) or while the loop runs; anywhere else it is stored for the next Home
  $("bal-trim-where").textContent = BAL.active ? "live (loop)"
    : b.stand_hold ? "live (holding the stand)" : "saved — applies at the next 🏠 Home";
  BAL.defaults = b.defaults || BAL.defaults;
  for (const el of document.querySelectorAll("[data-gain]")) {
    const v = (b.gains || {})[el.dataset.gain];
    if (v !== undefined && document.activeElement !== el && !el.dataset.dirty) el.value = v;
    const d = (BAL.defaults || {})[el.dataset.gain];
    el.classList.toggle("override", d !== undefined && Math.abs(+el.value - d) > 1e-9);
  }
}
$("btn-balance").onclick = async () => {
  if (BAL.active) {
    await api("/api/balance/stop", { method: "POST" });
    setBanner("balance stopped — holding the pose", "", 3000);
  } else {
    await api("/api/balance/start", { json: {} });
    setBanner("⚖ balancing — let go gently; Stop holds the pose, E-STOP goes limp", "warn", 5000);
  }
};
let balTrimTimer = null;
let balTrimSeq = 0;
const BAL_TRIM_INPUTS = [["bal-comx", "com_x_mm"], ["bal-comy", "com_y_mm"], ["bal-pitch", "pitch_deg"]];
function sendBalanceTrim() {
  clearTimeout(balTrimTimer);
  for (const [id] of BAL_TRIM_INPUTS) {          // ours until the daemon answers THIS edit: an
    const el = $(id);                            // older edit's echo must not release the guard
    el.dataset.dirty = "1"; delete el.dataset.want; delete el.dataset.wantT;
  }
  const seq = ++balTrimSeq;
  balTrimTimer = setTimeout(async () => {
    let d = null;
    try {
      d = await api("/api/balance/trim", { json: {
        com_x_mm: +$("bal-comx").value, com_y_mm: +$("bal-comy").value,
        pitch_deg: +$("bal-pitch").value } });
    } catch (_) { /* banner already set */ }
    if (seq !== balTrimSeq) return;          // a newer edit is on its way: leave its value alone
    for (const [id, key] of BAL_TRIM_INPUTS) {
      const el = $(id);
      if (d && d.trim && d.trim[key] !== undefined) {
        el.value = d.trim[key];                                   // as the daemon clipped it
        el.dataset.want = String(d.trim[key]);                    // dirty until a poll echoes it
        el.dataset.wantT = String(Date.now());
      } else {
        delete el.dataset.dirty;                                  // refused: show the daemon's copy
      }
    }
    $("bal-comx-val").textContent = (+$("bal-comx").value >= 0 ? "+" : "") + $("bal-comx").value;
    $("bal-comy-val").textContent = (+$("bal-comy").value >= 0 ? "+" : "") + $("bal-comy").value;
  }, 120);
}
for (const id of ["bal-comx", "bal-comy"]) {
  $(id).oninput = () => {
    $(id + "-val").textContent = (+$(id).value >= 0 ? "+" : "") + $(id).value;
    sendBalanceTrim();
  };
}
$("bal-pitch").onchange = sendBalanceTrim;
// ◀ ▶: one step of the slider (1 mm) or 0.1° of pitch per click, clamped to the input's range
for (const b of document.querySelectorAll("[data-bal-step]")) {
  b.onclick = (e) => {
    e.preventDefault();
    const el = $(b.dataset.balStep);
    const v = Math.min(+el.max, Math.max(+el.min, Math.round((+el.value + +b.dataset.d) * 10) / 10));
    el.value = v;
    if (el.oninput) el.oninput(); else sendBalanceTrim();
  };
}
async function sendGains(g) {
  try { await api("/api/balance/gains", { json: g }); } catch (_) { /* banner already set */ }
  for (const el of document.querySelectorAll("[data-gain]")) delete el.dataset.dirty;
}
for (const el of document.querySelectorAll("[data-gain]")) {
  el.oninput = () => { el.dataset.dirty = "1"; };
  el.onchange = () => { if (el.value !== "") sendGains({ [el.dataset.gain]: +el.value }); };
}
$("btn-gains-default").onclick = () => { if (BAL.defaults) sendGains(BAL.defaults); };
$("chk-override").onchange = () => {
  if ($("chk-override").checked &&
      !confirm("Override the safe-workspace check?\nOnly the physical assembly-band net remains.")) {
    $("chk-override").checked = false; return;
  }
  api("/api/manual", { json: { override: $("chk-override").checked } }).catch(() => {});
};

/* ================================================================ grid view (pan/zoom canvas) */
class GridView {
  constructor(canvas, coordsEl) {
    this.cv = canvas; this.g = canvas.getContext("2d"); this.coordsEl = coordsEl;
    this.scale = 4; this.ox = 0; this.oy = 0;  // world(deg) -> px: x' = (x-ox)*scale
    this.pointers = new Map();
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const r = this.cv.getBoundingClientRect();
      const px = (e.clientX - r.left) * this.cv.width / r.width;
      const py = (e.clientY - r.top) * this.cv.height / r.height;
      const w = this.toWorld(px, py);
      this.scale *= f;
      this.ox = w[0] - px / this.scale;
      this.oy = w[1] + py / this.scale;
      this.render();
    }, { passive: false });
  }
  fit(xmin, xmax, ymin, ymax, pad = 0.08) {
    const dx = (xmax - xmin) || 1, dy = (ymax - ymin) || 1;
    xmin -= dx * pad; xmax += dx * pad; ymin -= dy * pad; ymax += dy * pad;
    this.scale = Math.min(this.cv.width / (xmax - xmin), this.cv.height / (ymax - ymin));
    this.ox = xmin - (this.cv.width / this.scale - (xmax - xmin)) / 2;
    this.oy = ymax + (this.cv.height / this.scale - (ymax - ymin)) / 2;
  }
  toPx(x, y) { return [(x - this.ox) * this.scale, (this.oy - y) * this.scale]; }
  toWorld(px, py) { return [this.ox + px / this.scale, this.oy - py / this.scale]; }
  eventPx(e) {
    const r = this.cv.getBoundingClientRect();
    return [(e.clientX - r.left) * this.cv.width / r.width,
            (e.clientY - r.top) * this.cv.height / r.height];
  }
  drawAxes() {
    const g = this.g;
    g.lineWidth = 1;
    const [x0px, y0px] = this.toPx(0, 0);
    g.strokeStyle = "rgba(139,151,168,.35)";
    g.beginPath(); g.moveTo(x0px, 0); g.lineTo(x0px, this.cv.height); g.stroke();
    g.beginPath(); g.moveTo(0, y0px); g.lineTo(this.cv.width, y0px); g.stroke();
    // tick labels every ~50px
    g.fillStyle = "rgba(139,151,168,.7)"; g.font = "10px monospace";
    const stepDeg = niceStep(50 / this.scale);
    const wx0 = Math.floor(this.ox / stepDeg) * stepDeg;
    for (let x = wx0; x < this.ox + this.cv.width / this.scale; x += stepDeg) {
      const [px] = this.toPx(x, 0);
      g.fillText(x.toFixed(0), px + 2, this.cv.height - 4);
    }
    const wy0 = Math.floor((this.oy - this.cv.height / this.scale) / stepDeg) * stepDeg;
    for (let y = wy0; y < this.oy; y += stepDeg) {
      const [, py] = this.toPx(0, y);
      g.fillText(y.toFixed(0), 4, py - 2);
    }
  }
  render() {}   // overridden
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
  return 10 * p;
}

/* ================================================================ workspace editor */
const wsEd = {
  view: null, grid: null, shape: null, camO: 0, thighO: 0, res: 1,
  undo: [], redo: [], tool: "pan", dirty: false, lastCell: null,
  world: [0, 0],        // last world point the hit test saw (shift-invariant across a grow)
  prevWorld: [0, 0],    // ... and the one the current stroke came from
};

/* Put the live canvas span into the extent boxes (they are also the resize input). */
function wsSyncExtentInputs() {
  const on = !!(wsEd.grid && wsEd.shape);
  const [nc, nt] = on ? wsEd.shape : [0, 0];
  const vals = on
    ? [wsEd.camO, wsEd.camO + nc * wsEd.res, wsEd.thighO, wsEd.thighO + nt * wsEd.res]
    : ["", "", "", ""];
  ["ws-cam-lo", "ws-cam-hi", "ws-th-lo", "ws-th-hi"].forEach((id, k) => {
    const el = $(id);
    if (!el) return;
    el.disabled = !on;
    el.value = on ? Number(vals[k]).toFixed(1) : "";
  });
  const hint = $("ws-extent-hint");
  if (hint) {
    const [camLo, camHi] = jointLimit(S.wsLeg, "cam", "hard");
    const [thLo, thHi] = jointLimit(S.wsLeg, "thigh", "hard");
    hint.textContent = `limit cam ${camLo.toFixed(0)}…${camHi.toFixed(0)}°, ` +
                       `thigh ${thLo.toFixed(0)}…${thHi.toFixed(0)}°`;
  }
}

/* Resize to the typed span. Shrinking is allowed and DISCARDS the cells outside it -- undo holds
   the previous frame, and the stats line reports the new occupied count either way. */
function wsResizeFromInputs() {
  if (!wsEd.grid) return;
  const want = [+$("ws-cam-lo").value, +$("ws-cam-hi").value,
                +$("ws-th-lo").value, +$("ws-th-hi").value];
  if (want.some((v) => !isFinite(v))) { setBanner("span must be four numbers", "error", 3000); return; }
  if (want[0] >= want[1] || want[2] >= want[3]) {
    setBanner("each span must run low → high", "error", 3000); return;
  }
  const [camLo, camHi] = jointLimit(S.wsLeg, "cam", "hard");
  const [thLo, thHi] = jointLimit(S.wsLeg, "thigh", "hard");
  if (want[0] < camLo || want[1] > camHi || want[2] < thLo || want[3] > thHi) {
    setBanner(`span must stay inside the never-exceed range ` +
              `(cam ${camLo.toFixed(0)}…${camHi.toFixed(0)}°, ` +
              `thigh ${thLo.toFixed(0)}…${thHi.toFixed(0)}°)`, "error", 5000);
    return;
  }
  const r = wsEd.res, [nc, nt] = wsEd.shape;
  const pad = [Math.round((wsEd.camO - want[0]) / r),
               Math.round((want[1] - (wsEd.camO + nc * r)) / r),
               Math.round((wsEd.thighO - want[2]) / r),
               Math.round((want[3] - (wsEd.thighO + nt * r)) / r)];
  pushUndo();
  if (!wsResize(...pad)) { wsEd.undo.pop(); return; }
  wsSyncExtentInputs();
  wsEd.view.render();
  updateWsStats();
}

/* Pad (positive) or crop (negative) each edge by a number of cells. */
function wsResize(camLo, camHi, thLo, thHi) {
  const [nc, nt] = wsEd.shape;
  const nc2 = nc + camLo + camHi, nt2 = nt + thLo + thHi;
  if (nc2 < 1 || nt2 < 1) { setBanner("that span is smaller than one cell", "error", 3000); return false; }
  if (nc2 * nt2 > 4000000) { setBanner("that span is too many cells", "error", 3000); return false; }
  const g2 = new Uint8Array(nc2 * nt2);
  for (let i = 0; i < nc; i++) {
    const i2 = i + camLo;
    if (i2 < 0 || i2 >= nc2) continue;
    for (let j = 0; j < nt; j++) {
      const j2 = j + thLo;
      if (j2 < 0 || j2 >= nt2) continue;
      if (wsEd.grid[i * nt + j]) g2[i2 * nt2 + j2] = 1;
    }
  }
  wsEd.grid = g2; wsEd.shape = [nc2, nt2];
  wsEd.camO -= camLo * wsEd.res; wsEd.thighO -= thLo * wsEd.res;
  wsEd.dirty = true;
  return true;
}

function wsLegData() { return S.ws && S.ws.legs ? S.ws.legs[S.wsLeg] : null; }

function loadWsIntoEditor() {
  if (!wsEd.view) return;  // guard against early calls before canvas setup
  const d = wsLegData();
  if (!d) {
    wsEd.grid = null;
    wsEd.shape = null;
    wsEd.view.fit(-60, 60, -60, 60);  // use default bounds
    wsEd.view.render();
    drawAbd();
    updateWsStats();
    return;
  }
  const k = d.knee;
  if (!k || !k.shape) {
    wsEd.grid = null;
    wsEd.shape = null;
    wsEd.view.fit(-60, 60, -60, 60);
    wsEd.view.render();
    drawAbd();
    updateWsStats();
    return;
  }
  wsEd.shape = k.shape; wsEd.camO = k.cam_origin; wsEd.thighO = k.thigh_origin; wsEd.res = k.res_deg;
  wsEd.grid = unpackBits(k.grid_b64, k.shape[0] * k.shape[1]);
  wsEd.undo = []; wsEd.redo = []; wsEd.dirty = false;
  wsEd.view.fit(k.cam_origin, k.cam_origin + k.shape[0] * k.res_deg,
                k.thigh_origin, k.thigh_origin + k.shape[1] * k.res_deg);
  wsSyncExtentInputs();
  wsEd.view.render();
  drawAbd();
  updateWsStats();
  $("abd-min").value = d.abd_safe[0]; $("abd-max").value = d.abd_safe[1];
}

function unpackBits(b64, count) {
  const bin = atob(b64);
  const out = new Uint8Array(count);
  for (let i = 0; i < count; i++) out[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
  return out;
}
function packBits(arr) {
  const bytes = new Uint8Array(Math.ceil(arr.length / 8));
  for (let i = 0; i < arr.length; i++) if (arr[i]) bytes[i >> 3] |= 128 >> (i & 7);
  let s = "";
  for (let i = 0; i < bytes.length; i += 4096) s += String.fromCharCode(...bytes.subarray(i, i + 4096));
  return btoa(s);
}

function renderWs() {
  if (!wsEd.view || !wsEd.view.g) return;
  const v = wsEd.view, g = v.g, cv = v.cv;
  if (cv.width === 0 || cv.height === 0) return;  // canvas not visible
  g.fillStyle = "#10141a"; g.fillRect(0, 0, cv.width, cv.height);
  v.drawAxes();
  const d = wsLegData();
  if (wsEd.grid && wsEd.shape) {
    const [nc, nt] = wsEd.shape;
    // draw cells as rects (cheap enough <= ~25k cells; skip subpixel cells via batching)
    g.fillStyle = COLORS.region;
    const cell = wsEd.res * v.scale;
    for (let i = 0; i < nc; i++) {
      const x = (wsEd.camO + i * wsEd.res - v.ox) * v.scale;
      if (x < -cell || x > cv.width) continue;
      for (let j = 0; j < nt; j++) {
        if (!wsEd.grid[i * nt + j]) continue;
        const y = (v.oy - (wsEd.thighO + (j + 1) * wsEd.res)) * v.scale;
        if (y < -cell || y > cv.height) continue;
        g.fillRect(x, y, Math.max(cell, 1), Math.max(cell, 1));
      }
    }
  }
  if (d && d.samples) {
    g.fillStyle = COLORS.samples;
    for (const p of d.samples) {
      const [x, y] = v.toPx(p[0], p[1]);
      g.fillRect(x - 1, y - 1, 2, 2);
    }
  }
  // live sweep trail (accumulates AS YOU MOVE while recording a workspace pass)
  if (S.wsTrail.length > 1) {
    g.strokeStyle = COLORS.stroke; g.lineWidth = 2; g.beginPath();
    S.wsTrail.forEach((p, i) => { const [x, y] = v.toPx(p[0], p[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke();
    const [hx, hy] = v.toPx(...S.wsTrail[S.wsTrail.length - 1]);
    g.fillStyle = COLORS.stroke; g.beginPath(); g.arc(hx, hy, 3.5, 0, 7); g.fill();
  }
  // gait path of the shown trajectory
  const tr = S.traj && S.traj[S.wsLeg];
  if (tr && tr.path) drawLoop(g, v, tr.path, COLORS.gait, 2);
  // zero marker (0,0)
  const [zx, zy] = v.toPx(0, 0);
  g.strokeStyle = COLORS.zero; g.lineWidth = 1.6;
  g.strokeRect(zx - 5, zy - 5, 10, 10);
  // live crosshair
  const cam = S.latest[S.wsLeg + ".cam"], th = S.latest[S.wsLeg + ".thigh"];
  if (cam && th && cam.pos_norm !== null) {
    const [x, y] = v.toPx(cam.pos_norm, th.pos_norm);
    g.strokeStyle = "#4da3ff"; g.lineWidth = 1.4;
    g.beginPath(); g.moveTo(x - 10, y); g.lineTo(x + 10, y);
    g.moveTo(x, y - 10); g.lineTo(x, y + 10); g.stroke();
  }
  g.fillStyle = "rgba(215,222,232,.8)"; g.font = "11px monospace";
  g.fillText("cam° →", cv.width - 54, cv.height - 18);
  g.save(); g.translate(12, 60); g.rotate(-Math.PI / 2); g.fillText("thigh° →", 0, 0); g.restore();
}

function drawLoop(g, v, pts, color, lw) {
  g.strokeStyle = color; g.lineWidth = lw; g.beginPath();
  pts.forEach((p, i) => {
    const [x, y] = v.toPx(p[0], p[1]);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.closePath(); g.stroke();
}

function updateWsStats() {
  if (!wsEd.grid) {
    $("ws-stats").textContent = "no workspace for this leg yet — import or record a sweep";
    wsSyncExtentInputs();
    return;
  }
  let n = 0; for (let i = 0; i < wsEd.grid.length; i++) n += wsEd.grid[i];
  const [nc, nt] = wsEd.shape, r = wsEd.res;
  const span = `canvas cam ${wsEd.camO.toFixed(0)}…${(wsEd.camO + nc * r).toFixed(0)}°, ` +
               `thigh ${wsEd.thighO.toFixed(0)}…${(wsEd.thighO + nt * r).toFixed(0)}°`;
  $("ws-stats").innerHTML = `${n} / ${wsEd.grid.length} cells safe (${r}°/cell) — ${span}` +
    (n === 0 ? ' — <b style="color:#e04545">EMPTY: nothing will pass the safety check!</b>' : "") +
    (wsEd.dirty ? ' — <b style="color:#e0a020">unapplied edits</b>' : "");
}

/* The grid's extent used to be frozen at whatever the backdriven sweep happened to cover: the
   array is built as [min(sample)-1deg, max(sample)+1deg] (calibrate_workspace._knee_grid), and
   this hit test returned null outside it, so a brush stroke past the edge was silently dropped.
   That read as a hard limit -- "I cannot draw past 36 deg of thigh" -- when 36 was simply where
   that operator's sweep had stopped, with the joint itself good for a great deal more.

   So the canvas grows. Drawing past an edge pads the array out to the cell under the cursor,
   bounded by the joint's never-exceed range; a padded cell is empty until it is painted, and
   the daemon sizes its clamps off the OCCUPIED cells, so growing the canvas alone widens nothing.
   Pan and erase never grow: only a stroke that is trying to ADD area. */
function wsWorldToCell(wx, wy) {
  return [Math.floor((wx - wsEd.camO) / wsEd.res), Math.floor((wy - wsEd.thighO) / wsEd.res)];
}

function wsGrow(padCamLo, padCamHi, padThLo, padThHi) {
  if (!(padCamLo || padCamHi || padThLo || padThHi)) return false;
  const [nc, nt] = wsEd.shape;
  const nc2 = nc + padCamLo + padCamHi, nt2 = nt + padThLo + padThHi;
  const g2 = new Uint8Array(nc2 * nt2);
  for (let i = 0; i < nc; i++) {
    const src = i * nt, dst = (i + padCamLo) * nt2 + padThLo;
    for (let j = 0; j < nt; j++) if (wsEd.grid[src + j]) g2[dst + j] = 1;
  }
  wsEd.grid = g2;
  wsEd.shape = [nc2, nt2];
  wsEd.camO -= padCamLo * wsEd.res;
  wsEd.thighO -= padThLo * wsEd.res;
  wsEd.dirty = true;
  return true;
}

/* How many cells may be added on each side before the span leaves the never-exceed range. */
function wsGrowRoom() {
  const [camLo, camHi] = jointLimit(S.wsLeg, "cam", "hard");
  const [thLo, thHi] = jointLimit(S.wsLeg, "thigh", "hard");
  const [nc, nt] = wsEd.shape, r = wsEd.res;
  return {
    camLo: Math.max(0, Math.floor((wsEd.camO - camLo) / r)),
    camHi: Math.max(0, Math.floor((camHi - (wsEd.camO + nc * r)) / r)),
    thLo: Math.max(0, Math.floor((wsEd.thighO - thLo) / r)),
    thHi: Math.max(0, Math.floor((thHi - (wsEd.thighO + nt * r)) / r)),
  };
}

/* Grid cell under a world point. `grow` asks for the canvas to be extended to reach it. Returns
   null when the point is outside the grid and cannot (or may not) be reached. */
function wsCellAt(wx, wy, grow) {
  if (!wsEd.grid || !wsEd.shape) return null;
  wsEd.world = [wx, wy];                       // world coords survive a reindexing; cells do not
  let [i, j] = wsWorldToCell(wx, wy);
  const [nc, nt] = wsEd.shape;
  const outside = i < 0 || i >= nc || j < 0 || j >= nt;
  if (outside && grow) {
    const room = wsGrowRoom();
    const pad = [Math.min(Math.max(0, -i), room.camLo),
                 Math.min(Math.max(0, i - nc + 1), room.camHi),
                 Math.min(Math.max(0, -j), room.thLo),
                 Math.min(Math.max(0, j - nt + 1), room.thHi)];
    if (wsGrow(...pad)) {
      const blocked = (i < -pad[0]) || (i > nc - 1 + pad[1]) ||
                      (j < -pad[2]) || (j > nt - 1 + pad[3]);
      [i, j] = wsWorldToCell(wx, wy);
      wsSyncExtentInputs();
      if (blocked) wsFlagAtLimit();
    } else {
      wsFlagAtLimit();
    }
  }
  const [nc2, nt2] = wsEd.shape;
  return (i >= 0 && i < nc2 && j >= 0 && j < nt2) ? [i, j] : null;
}

function wsFlagAtLimit() {
  const [camLo, camHi] = jointLimit(S.wsLeg, "cam", "hard");
  const [thLo, thHi] = jointLimit(S.wsLeg, "thigh", "hard");
  setBanner(`that is past the never-exceed range for this joint ` +
            `(cam ${camLo.toFixed(0)}…${camHi.toFixed(0)}°, ` +
            `thigh ${thLo.toFixed(0)}…${thHi.toFixed(0)}°) — the canvas stops there`, "", 3500);
}

function wsApplyBrush(cell, value) {
  const size = +$("brush-size").value, r = (size - 1) / 2;
  const [nc, nt] = wsEd.shape;
  for (let di = -r; di <= r; di++)
    for (let dj = -r; dj <= r; dj++) {
      const i = cell[0] + di, j = cell[1] + dj;
      if (i >= 0 && i < nc && j >= 0 && j < nt) wsEd.grid[i * nt + j] = value;
    }
  wsEd.dirty = true;
}

function wsFloodFill(cell) {
  const [nc, nt] = wsEd.shape;
  const start = wsEd.grid[cell[0] * nt + cell[1]];
  const target = start ? 0 : 1;
  const stack = [cell];
  const seen = new Uint8Array(nc * nt);
  let guard = 0;
  const guardMax = 5 * nc * nt;
  while (stack.length && guard++ < guardMax) {
    const [i, j] = stack.pop();
    const idx = i * nt + j;
    if (seen[idx] || wsEd.grid[idx] !== start) continue;
    seen[idx] = 1; wsEd.grid[idx] = target;
    if (i > 0) stack.push([i - 1, j]);
    if (i < nc - 1) stack.push([i + 1, j]);
    if (j > 0) stack.push([i, j - 1]);
    if (j < nt - 1) stack.push([i, j + 1]);
  }
  wsEd.dirty = true;
}

/* A snapshot is the grid AND its frame: the canvas can be resized now, so restoring cells into a
   different shape or origin would put them somewhere else entirely. */
function wsSnap() {
  return { grid: wsEd.grid.slice(), shape: wsEd.shape.slice(),
           camO: wsEd.camO, thighO: wsEd.thighO, res: wsEd.res };
}
function wsRestore(snap) {
  wsEd.grid = snap.grid; wsEd.shape = snap.shape;
  wsEd.camO = snap.camO; wsEd.thighO = snap.thighO; wsEd.res = snap.res;
  wsEd.dirty = true;
  wsSyncExtentInputs();
}
function pushUndo() {
  wsEd.undo.push(wsSnap());
  if (wsEd.undo.length > 50) wsEd.undo.shift();
  wsEd.redo = [];
}

function setupWsCanvas() {
  const v = new GridView($("ws-canvas"), $("ws-coords"));
  v.render = renderWs;
  wsEd.view = v;
  attachPanZoomDraw(v, () => wsEd.tool, {
    onStrokeStart: (cell) => { if (!wsEd.grid) return;
      pushUndo();
      wsEd.prevWorld = wsEd.world.slice();
      if (wsEd.tool === "fill") { wsFloodFill(cell); v.render(); updateWsStats(); }
      else { wsApplyBrush(cell, wsEd.tool === "draw" ? 1 : 0); v.render(); }
    },
    onStrokeMove: (cell) => { if (!wsEd.grid || wsEd.tool === "fill") return;
      // Interpolate between events so a fast stroke does not gap. The previous point is carried in
      // WORLD degrees, not as a cell index: growing the canvas mid-stroke renumbers every cell, and
      // an index captured before the growth would smear the brush across the shift.
      const prev = wsWorldToCell(...wsEd.prevWorld);
      const steps = Math.max(Math.abs(cell[0] - prev[0]), Math.abs(cell[1] - prev[1]), 1);
      for (let s = 1; s <= steps; s++) {
        const i = Math.round(prev[0] + (cell[0] - prev[0]) * s / steps);
        const j = Math.round(prev[1] + (cell[1] - prev[1]) * s / steps);
        wsApplyBrush([i, j], wsEd.tool === "draw" ? 1 : 0);
      }
      wsEd.prevWorld = wsEd.world.slice();
      v.render();
    },
    onStrokeEnd: () => updateWsStats(),
    // only a stroke that ADDS area may grow the canvas; pan, erase and fill stay inside it
    cellAt: (wx, wy) => wsCellAt(wx, wy, wsEd.tool === "draw"),
  });
  $("btn-ws-resize").onclick = () => wsResizeFromInputs();
  $("btn-ws-grow-max").onclick = () => {
    if (!wsEd.grid) return;
    pushUndo();
    const room = wsGrowRoom();
    if (!wsGrow(room.camLo, room.camHi, room.thLo, room.thHi)) {
      wsEd.undo.pop();
      setBanner("the canvas already spans the joint's full never-exceed range", "", 2500);
      return;
    }
    wsSyncExtentInputs(); v.render(); updateWsStats();
  };
  $("ws-toolbar").querySelectorAll(".tool").forEach((b) => b.onclick = () => {
    $("ws-toolbar").querySelectorAll(".tool").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    wsEd.tool = b.dataset.tool;
  });
  $("btn-undo").onclick = () => { if (wsEd.undo.length) { wsEd.redo.push(wsSnap()); wsRestore(wsEd.undo.pop()); v.render(); updateWsStats(); } };
  $("btn-redo").onclick = () => { if (wsEd.redo.length) { wsEd.undo.push(wsSnap()); wsRestore(wsEd.redo.pop()); v.render(); updateWsStats(); } };
  $("btn-ws-apply").onclick = async () => {
    if (!wsEd.grid) return;
    await api("/api/workspace/grid", { json: { leg: S.wsLeg, grid_b64: packBits(wsEd.grid),
      shape: wsEd.shape, cam_origin: wsEd.camO, thigh_origin: wsEd.thighO, res_deg: wsEd.res } });
    wsEd.dirty = false;
    await refreshWorkspace();
    setBanner("workspace applied to the live safety check", "", 2500);
  };
  $("btn-ws-revert").onclick = () => loadWsIntoEditor();
}

/* generic pointer handling: pan/zoom always available (pan tool or 2-finger / middle button),
   draw tools call the stroke callbacks with grid cells */
function attachPanZoomDraw(view, getTool, cb) {
  const cv = view.cv;
  let panning = null, stroking = false, lastCell = null, pinch = null;
  cv.addEventListener("pointerdown", (e) => {
    cv.setPointerCapture(e.pointerId);
    view.pointers.set(e.pointerId, view.eventPx(e));
    if (view.pointers.size === 2) { pinch = pinchState(view); stroking = false; return; }
    const tool = getTool();
    const px = view.eventPx(e);
    if (tool === "pan" || e.button === 1 || e.button === 2) {
      panning = { px, ox: view.ox, oy: view.oy };
    } else if (cb.cellAt) {
      const cell = cb.cellAt(...view.toWorld(...px));
      if (cell) { stroking = true; lastCell = cell; cb.onStrokeStart(cell); }
    }
  });
  cv.addEventListener("pointermove", (e) => {
    const px = view.eventPx(e);
    if (view.pointers.has(e.pointerId)) view.pointers.set(e.pointerId, px);
    if (pinch && view.pointers.size === 2) { applyPinch(view, pinch); return; }
    const w = view.toWorld(...px);
    if (view.coordsEl) view.coordsEl.textContent = `cam ${w[0].toFixed(1)}°, thigh ${w[1].toFixed(1)}°`;
    if (panning) {
      view.ox = panning.ox - (px[0] - panning.px[0]) / view.scale;
      view.oy = panning.oy + (px[1] - panning.px[1]) / view.scale;
      view.render();
    } else if (stroking) {
      const cell = cb.cellAt(...w);
      if (cell && lastCell && (cell[0] !== lastCell[0] || cell[1] !== lastCell[1])) {
        cb.onStrokeMove(cell, lastCell); lastCell = cell;
      }
    }
  });
  const up = (e) => {
    view.pointers.delete(e.pointerId);
    if (view.pointers.size < 2) pinch = null;
    if (stroking) { stroking = false; cb.onStrokeEnd && cb.onStrokeEnd(); }
    panning = null;
  };
  cv.addEventListener("pointerup", up);
  cv.addEventListener("pointercancel", up);
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
}
function pinchState(view) {
  const [a, b] = [...view.pointers.values()];
  return { d: Math.hypot(a[0] - b[0], a[1] - b[1]), scale: view.scale,
    mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], ox: view.ox, oy: view.oy };
}
function applyPinch(view, p) {
  const [a, b] = [...view.pointers.values()];
  const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
  const f = d / p.d;
  const w = [p.ox + p.mid[0] / p.scale, p.oy - p.mid[1] / p.scale];
  view.scale = p.scale * f;
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  view.ox = w[0] - mid[0] / view.scale;
  view.oy = w[1] + mid[1] / view.scale;
  view.render();
}

/* ---------------- abduction bar ---------------- */
function drawAbd() {
  const cv = $("ws-abd-canvas"), g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  const d = wsLegData();
  if (!d) return;
  const lo = Math.min(d.abd_observed[0], -5), hi = Math.max(d.abd_observed[1], 5);
  const pad = (hi - lo) * 0.1;
  const X = (v) => ((v - lo + pad) / (hi - lo + 2 * pad)) * cv.width;
  const y = cv.height / 2;
  g.strokeStyle = "#5a6474"; g.lineWidth = 12; g.lineCap = "round";
  g.beginPath(); g.moveTo(X(d.abd_observed[0]), y); g.lineTo(X(d.abd_observed[1]), y); g.stroke();
  g.strokeStyle = COLORS.good;
  g.beginPath(); g.moveTo(X(d.abd_safe[0]), y); g.lineTo(X(d.abd_safe[1]), y); g.stroke();
  // live abduction swept so far during the current sweep (thin yellow overlay above the bar)
  if (S.wsAbdSweep[0] <= S.wsAbdSweep[1]) {
    g.strokeStyle = COLORS.stroke; g.lineWidth = 4;
    g.beginPath(); g.moveTo(X(S.wsAbdSweep[0]), y - 12); g.lineTo(X(S.wsAbdSweep[1]), y - 12); g.stroke();
  }
  g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.setLineDash([4, 3]);
  g.beginPath(); g.moveTo(X(0), 6); g.lineTo(X(0), cv.height - 6); g.stroke(); g.setLineDash([]);
  const live = S.latest[S.wsLeg + ".abd"];
  if (live && live.pos_norm !== null) {
    g.fillStyle = "#4da3ff";
    g.beginPath(); g.arc(X(live.pos_norm), y, 6, 0, 7); g.fill();
  }
  g.fillStyle = "rgba(139,151,168,.9)"; g.font = "10px monospace";
  g.fillText(`${d.abd_observed[0].toFixed(1)}°`, 2, y + 22);
  g.fillText(`${d.abd_observed[1].toFixed(1)}°`, cv.width - 44, y + 22);
  g.fillText("zero", X(0) - 12, 12);
}
$("btn-abd-apply").onclick = () => api("/api/workspace/abduction", { json: {
  leg: S.wsLeg, safe_min: +$("abd-min").value, safe_max: +$("abd-max").value } })
  .then(refreshWorkspace);

/* ---------------- workspace files / recording ---------------- */
async function refreshWorkspace() {
  try {
    const r = await fetch("/api/workspace");
    if (!r.ok) {
      console.error("workspace fetch failed:", r.status, r.statusText);
      return;
    }
    S.ws = await r.json();
    if (!S.ws) {
      console.error("workspace response is null");
      return;
    }
    $("ws-source").textContent = S.ws.source ? "· " + S.ws.source : "";
    const legs = (S.ws && S.ws.legs) || {};
    legBadges($("ws-legs-loaded"), !!legs.right, !!legs.left);
    legBadges($("ws-legs-loaded2"), !!legs.right, !!legs.left);
    loadWsIntoEditor();
    updateManualRanges();
    renderEE();
  } catch (e) {
    console.error("refreshWorkspace error:", e);
    setBanner("Failed to refresh workspace: " + e.message, "error", 5000);
  }
}

/* Slider bounds follow the daemon's NOMINAL range: the CAD guess widened by the demonstrated
   workspace. Deliberately not the never-exceed clamp -- that is +-180 deg on the knee pair and a
   slider spanning it would put a hard-stop collision one careless drag away. */
function updateManualRanges() {
  for (const n of MOTORS) {
    const [side, role] = n.split(".");
    const [lo, hi] = jointLimit(side, role, "nominal");
    const row = $("man-" + n.replace(".", "-"));
    row.querySelector(".mr-slider").min = lo;
    row.querySelector(".mr-slider").max = hi;
  }
}
$("ws-tabs").querySelectorAll(".tab").forEach((b) => b.onclick = async () => {
  try {
    $("ws-tabs").querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    S.wsLeg = b.dataset.leg;
    $("btn-ws-mirror").textContent =
      S.wsLeg === "right" ? "⇄ copy right → left" : "⇄ copy left → right";
    document.querySelectorAll(".wsrec-legname").forEach((s) => s.textContent = S.wsLeg);
    resetWsTrail();
    await refreshWorkspace();  // ensure we have fresh data for the new leg
    // loadWsIntoEditor is called by refreshWorkspace, but force render to be safe
    wsEd.view.render();
  } catch (e) {
    console.error("workspace tab switch error:", e);
    setBanner("Error switching workspace leg: " + e.message, "error", 5000);
  }
});
function resetWsTrail() { S.wsTrail = []; S.wsAbdSweep = [Infinity, -Infinity]; }
$("btn-ws-mirror").onclick = async () => {
  const from = S.wsLeg, to = from === "right" ? "left" : "right";
  if (!confirm(`Overwrite the ${to} leg's workspace with a copy of ${from}?`)) return;
  await api("/api/workspace/mirror", { json: { from, to, flips: {
    abd: $("mir-abd").checked, cam: $("mir-cam").checked, thigh: $("mir-thigh").checked } } });
  setBanner(`workspace copied ${from} → ${to}`, "", 3000);
  await refreshWorkspace();
};
$("btn-ws-save").onclick = () => api("/api/workspace/save", { json: { name: $("ws-name").value } })
  .then(() => { setBanner("workspace saved", "", 2000); refreshWorkspace(); });
$("btn-ws-export").onclick = () => { window.location = "/api/workspace/export"; };
$("ws-import").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  const r = await fetch("/api/workspace/import", { method: "POST", body: fd });
  const d = await r.json();
  if (!d.ok) { setBanner(d.error, "error", 8000); return; }
  setBanner(d.message || "imported", "", 4000);
  e.target.value = "";
  await refreshWorkspace();
};
$("btn-ws-load").onclick = async () => {
  const name = $("ws-files").value;
  if (!name) return;
  const blob = await (await fetch(`/api/workspace/export?name=${encodeURIComponent(name)}`)).blob();
  const fd = new FormData();
  fd.append("file", new File([blob], name));
  await fetch("/api/workspace/import", { method: "POST", body: fd });
  await refreshWorkspace();
};

$("btn-wsrec-mode").onclick = () => { resetWsTrail(); api("/api/record/mode", { json: { kind: "workspace" } }); };
$("btn-wsrec-take").onclick = () => {
  const active = S.state.recording && S.state.recording.active;
  api("/api/record/take", { json: { leg: S.wsLeg, action: active ? "stop" : "start" } });
};
$("btn-wsrec-undo").onclick = () => api("/api/record/undo", { json: { leg: S.wsLeg } });
$("btn-wsrec-process").onclick = () => api("/api/workspace/process", { json: {
  leg: S.wsLeg, margin_deg: +$("wsrec-margin").value, grid_deg: +$("wsrec-grid").value,
  dilate_deg: +$("wsrec-dilate").value } }).then(() => {   // the built green region now stands in for the raw trail
    resetWsTrail(); refreshWorkspace();
    setBanner(`${S.wsLeg} workspace built from the sweep`, "", 2500);
  });

/* Accumulate the live (cam,thigh) trail + swept abduction range while a workspace pass is running,
 * so the plot fills in AS YOU MOVE. Cleared automatically once you leave sweep mode.
 * From EVERY sample of the telemetry batch `d`, not the last one: the daemon samples at 20 Hz
 * whatever the link does, but a poll over the hotspot can take seconds to arrive, and one point per
 * poll drew a straight line between wherever the leg was at each arrival (2026-09-22). The recorded
 * workspace was never affected -- the daemon stores every 100 Hz sample -- only this drawing. */
function accumulateWsTrail(d) {
  const st = S.state;
  if (!st || st.mode !== "RECORD_WS") { if (S.wsTrail.length) resetWsTrail(); return; }
  const rec = st.recording || {};
  if (!rec.active) return;                      // only extend while a pass is actually recording
  const leg = rec.leg || S.wsLeg;
  const cam = d && d.motors[leg + ".cam"], th = d && d.motors[leg + ".thigh"], ab = d && d.motors[leg + ".abd"];
  const n = (d && d.t) ? d.t.length : 0;
  if (!cam || !th || !n) return;
  for (let i = 0; i < n; i++) {
    if (cam.pos_norm[i] == null || th.pos_norm[i] == null) continue;
    const p = [cam.pos_norm[i], th.pos_norm[i]];
    const last = S.wsTrail[S.wsTrail.length - 1];
    if (!last || Math.hypot(p[0] - last[0], p[1] - last[1]) > 0.25) S.wsTrail.push(p);
    if (ab && ab.pos_norm[i] != null) {
      S.wsAbdSweep[0] = Math.min(S.wsAbdSweep[0], ab.pos_norm[i]);
      S.wsAbdSweep[1] = Math.max(S.wsAbdSweep[1], ab.pos_norm[i]);
    }
  }
}

/* ================================================================ trajectory panel */
const trEd = { view: null, stroke: [], tool: "pan" };

function setupTrajCanvas() {
  const v = new GridView($("traj-canvas"), $("traj-coords"));
  v.render = renderTraj;
  trEd.view = v;
  attachPanZoomDraw(v, () => trEd.tool, {
    cellAt: (wx, wy) => [wx, wy],           // free coordinates, not grid cells
    onStrokeStart: (w) => { trEd.stroke = [w]; v.render(); },
    onStrokeMove: (w) => { trEd.stroke.push(w); v.render(); },
    onStrokeEnd: () => { v.render(); updateTrajStats(); },
  });
  document.querySelectorAll("#panel-trajectory .tool").forEach((b) => b.onclick = () => {
    document.querySelectorAll("#panel-trajectory .tool").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    trEd.tool = b.dataset.ttool;
  });
}

const TRAJ_COLORS = { right: "#ff35c8", left: "#35d0ff" };
const trajBackdrops = {};                       // leg -> {key, grid} unpacked-bits cache

function previewPhase() {
  // floor matches daemon.PERIOD_MIN (0.4 s = 2.5 Hz hard limit), so the on-screen preview and the
  // robot stay in step
  const period = Math.max(0.4, +$("pb-period").value || 8);
  return (((performance.now() - S.preview.t0) / 1000) / period) % 1;
}
function previewIdx(tr, side, p) {
  // path[i] = reconstruct(phase=i/N) which bakes in the FILE's phase_shift; emulate playback
  // with the UI left-phase slider: canonical(p + L) -> i = N*(p + L - fileShift)
  const N = tr.path.length;
  let ph = p;
  if (side === "left")
    ph = p + (+$("pb-leftphase").value) - (tr.phase_shift !== undefined ? tr.phase_shift : 0.5);
  return Math.floor((((ph % 1) + 1) % 1) * N) % N;
}

function renderTraj() {
  if (!trEd.view || !trEd.view.g) return;
  const v = trEd.view, g = v.g, cv = v.cv;
  if (cv.width === 0 || cv.height === 0) return;  // canvas not visible
  g.fillStyle = "#10141a"; g.fillRect(0, 0, cv.width, cv.height);
  v.drawAxes();
  // workspace backdrops for BOTH legs (active leg green, other faded — the normalized frames
  // coincide, so overlap is meaningful)
  for (const leg of ["left", "right"]) {
    const d = S.ws && S.ws.legs ? S.ws.legs[leg] : null;
    if (!d) continue;
    const k = d.knee;
    const key = k.grid_b64;
    if (!trajBackdrops[leg] || trajBackdrops[leg].key !== key)
      trajBackdrops[leg] = { key, grid: unpackBits(k.grid_b64, k.shape[0] * k.shape[1]) };
    const grid = trajBackdrops[leg].grid;
    g.fillStyle = leg === S.trajLeg ? "rgba(44,158,63,0.30)" : "rgba(120,150,170,0.10)";
    const cell = k.res_deg * v.scale;
    for (let i = 0; i < k.shape[0]; i++) {
      const x = (k.cam_origin + i * k.res_deg - v.ox) * v.scale;
      if (x < -cell || x > cv.width) continue;
      for (let j = 0; j < k.shape[1]; j++) {
        if (!grid[i * k.shape[1] + j]) continue;
        const y = (v.oy - (k.thigh_origin + (j + 1) * k.res_deg)) * v.scale;
        if (y < -cell || y > cv.height) continue;
        g.fillRect(x, y, Math.max(cell, 1), Math.max(cell, 1));
      }
    }
  }
  // gait loops for BOTH legs (active leg emphasized)
  for (const leg of ["left", "right"]) {
    const tr = S.traj && S.traj[leg];
    if (!tr || !tr.path) continue;
    g.globalAlpha = leg === S.trajLeg ? 1 : 0.55;
    if (leg !== S.trajLeg) g.setLineDash([6, 4]);
    drawLoop(g, v, tr.path, TRAJ_COLORS[leg], leg === S.trajLeg ? 2.5 : 1.5);
    g.setLineDash([]);
    g.globalAlpha = 1;
  }
  if (trEd.stroke.length > 1) {
    g.strokeStyle = COLORS.stroke; g.lineWidth = 2; g.beginPath();
    trEd.stroke.forEach((p, i) => {
      const [x, y] = v.toPx(p[0], p[1]);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
    // direction arrow at 1/4 of the stroke
    const q = Math.floor(trEd.stroke.length / 4);
    if (q > 1) drawArrow(g, v, trEd.stroke[q - 1], trEd.stroke[q], COLORS.stroke);
  }
  const [zx, zy] = v.toPx(0, 0);
  g.strokeStyle = COLORS.zero; g.lineWidth = 1.6; g.strokeRect(zx - 5, zy - 5, 10, 10);
  // live crosshairs for both legs
  for (const leg of ["left", "right"]) {
    const cam = S.latest[leg + ".cam"], th = S.latest[leg + ".thigh"];
    if (!cam || !th || cam.pos_norm === null) continue;
    const [x, y] = v.toPx(cam.pos_norm, th.pos_norm);
    g.globalAlpha = leg === S.trajLeg ? 1 : 0.45;
    g.strokeStyle = "#4da3ff"; g.lineWidth = 1.4;
    g.beginPath(); g.moveTo(x - 10, y); g.lineTo(x + 10, y);
    g.moveTo(x, y - 10); g.lineTo(x, y + 10); g.stroke();
    g.globalAlpha = 1;
  }
  // preview: two markers running along the loops with the playback period + dephasing
  if (S.preview.on && S.traj) {
    const p = previewPhase();
    for (const leg of ["left", "right"]) {
      const tr = S.traj[leg];
      if (!tr || !tr.path) continue;
      const pt = tr.path[previewIdx(tr, leg, p)];
      const [x, y] = v.toPx(pt[0], pt[1]);
      g.fillStyle = TRAJ_COLORS[leg]; g.strokeStyle = "#fff"; g.lineWidth = 1.5;
      g.beginPath(); g.arc(x, y, 7, 0, 7); g.fill(); g.stroke();
    }
  }
  // legend
  g.font = "11px sans-serif";
  g.fillStyle = TRAJ_COLORS.right; g.fillText("● right gait", 8, 16);
  g.fillStyle = TRAJ_COLORS.left; g.fillText("● left gait", 8, 30);
}
function drawArrow(g, v, p0, p1, color) {
  const [x0, y0] = v.toPx(p0[0], p0[1]), [x1, y1] = v.toPx(p1[0], p1[1]);
  const a = Math.atan2(y1 - y0, x1 - x0);
  g.fillStyle = color; g.beginPath();
  g.moveTo(x1, y1);
  g.lineTo(x1 - 12 * Math.cos(a - 0.4), y1 - 12 * Math.sin(a - 0.4));
  g.lineTo(x1 - 12 * Math.cos(a + 0.4), y1 - 12 * Math.sin(a + 0.4));
  g.fill();
}

function updateTrajStats() {
  const n = trEd.stroke.length;
  $("traj-stats").textContent = n ? `stroke: ${n} points — "Use drawn path" smooths + closes it via the standard pipeline` : "";
}

$("traj-tabs").querySelectorAll(".tab").forEach((b) => b.onclick = async () => {
  try {
    $("traj-tabs").querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    S.trajLeg = b.dataset.leg;
    document.querySelectorAll(".traj-legname").forEach((s) => s.textContent = S.trajLeg);
    trEd.stroke = [];
    await refreshWorkspace();  // ensure workspace data is fresh for this leg
    fitTrajView();
    trEd.view.render();
  } catch (e) {
    console.error("trajectory tab switch error:", e);
    setBanner("Error switching trajectory leg: " + e.message, "error", 5000);
  }
});
function updateTrajLegBadges() {
  const t = S.traj || {};
  legBadges($("traj-legs-loaded"), !!t.right, !!t.left);
}
function fitTrajView() {
  if (!trEd.view) return;  // guard against early calls before canvas setup
  const d = S.ws && S.ws.legs ? S.ws.legs[S.trajLeg] : null;
  if (d && d.knee) {
    const k = d.knee;
    trEd.view.fit(k.cam_origin, k.cam_origin + k.shape[0] * k.res_deg,
                  k.thigh_origin, k.thigh_origin + k.shape[1] * k.res_deg);
  } else {
    trEd.view.fit(-60, 60, -60, 60);
  }
  trEd.view.render();  // ensure view renders after fitting
}
$("btn-traj-clear").onclick = () => { trEd.stroke = []; trEd.view.render(); updateTrajStats(); };
$("btn-traj-reverse").onclick = () => { trEd.stroke.reverse(); trEd.view.render(); };
$("btn-traj-usepath").onclick = async () => {
  if (trEd.stroke.length < 8) { setBanner("draw a path first (≥8 points)", "warn", 3000); return; }
  const abd = S.latest[S.trajLeg + ".abd"];
  const d = await api("/api/trajectory/draw", { json: {
    name: $("traj-name").value, leg: S.trajLeg,
    points: trEd.stroke.map((p) => [p[0], p[1]]),
    abd_hold: abd && abd.pos_norm !== null ? abd.pos_norm : 0 } });
  S.traj = d.trajectory; S.trajName = $("traj-name").value;
  updateTrajLegBadges();
  setBanner("drawn path processed + saved as " + $("traj-name").value, "", 3500);
  trEd.view.render(); wsEd.view.render(); renderEE();
};

function previewLoop() {
  if (!S.preview.on) return;
  trEd.view.render();
  renderEE();
  requestAnimationFrame(previewLoop);
}
$("btn-traj-preview").onclick = () => {
  S.preview.on = !S.preview.on;
  if (S.preview.on) {
    if (!S.traj) { setBanner("show or draw a trajectory first", "warn", 3000); S.preview.on = false; return; }
    S.preview.t0 = performance.now();
    previewLoop();
  }
  $("btn-traj-preview").textContent = S.preview.on ? "■ stop preview" : "▶ preview both legs";
  $("btn-traj-preview").classList.toggle("active-rec", S.preview.on);
};

$("btn-traj-copyleg").onclick = async () => {
  const name = S.trajName || $("traj-name").value;
  if (!name) { setBanner("show or save a trajectory first", "warn", 3000); return; }
  const from = S.trajLeg, to = from === "right" ? "left" : "right";
  if (!(S.traj && S.traj[from])) {
    setBanner(`the shown trajectory has no ${from}-leg data — draw or record it first`, "warn", 4000);
    return;
  }
  if (!confirm(`Copy the ${from}-leg gait onto the ${to} leg in "${name}"?\n(the ${to} leg plays ` +
               `${to === "left" ? "dephased by the left-phase value" : "at phase 0"})`)) return;
  const d = await api("/api/trajectory/mirror", { json: {
    name, from, to, left_phase: +$("rec-leftphase").value } });
  S.traj = d.trajectory; S.trajName = name;
  updateTrajLegBadges();
  setBanner(`gait copied ${from} → ${to}`, "", 3000);
  trEd.view.render(); renderEE();
};

/* ---------------- gait teach recording ---------------- */
$("btn-rec-mode").onclick = () => api("/api/record/mode", { json: { kind: "gait" } });
$("btn-rec-take").onclick = () => {
  const active = S.state.recording && S.state.recording.active;
  api("/api/record/take", { json: { leg: S.trajLeg, action: active ? "stop" : "start" } });
};
$("btn-rec-center").onclick = () => api("/api/record/center", { json: { leg: S.trajLeg } });
$("btn-rec-undo").onclick = () => api("/api/record/undo", { json: { leg: S.trajLeg } });
$("btn-rec-reset").onclick = () => api("/api/record/reset", { method: "POST" });
$("btn-rec-finish").onclick = async () => {
  const d = await api("/api/record/finish", { json: {
    name: $("rec-name").value, harmonics: +$("rec-harmonics").value,
    split: +$("rec-split").value, left_phase: +$("rec-leftphase").value } });
  S.traj = d.trajectory; S.trajName = $("rec-name").value;
  updateTrajLegBadges();
  setBanner("gait processed + saved as " + $("rec-name").value, "", 3500);
  trEd.view.render(); wsEd.view.render(); renderEE();
};

function updateRecordUI(st) {
  const r = st.recording || {};
  const inGait = st.mode === "RECORD_GAIT", inWs = st.mode === "RECORD_WS";

  // ---- gait teach ----
  const gTakes = r.takes || { right: 0, left: 0 };
  const gCtr = r.centers || { right: null, left: null };
  $("btn-rec-take").disabled = !inGait;
  $("btn-rec-center").disabled = !inGait;
  $("btn-rec-undo").disabled = !inGait || r.active || !gTakes[S.trajLeg];
  $("btn-rec-take").textContent = (inGait && r.active) ? `■ stop take (${r.n_samples})` : "▶ start take";
  $("btn-rec-take").classList.toggle("active-rec", inGait && r.active);
  const legWord = (leg) => leg === S.trajLeg ? `[${leg}]` : leg;      // bracket the leg you're on
  $("rec-status").textContent = inGait ?
    `${legWord("right")}: ${gTakes.right || "no"} take${gTakes.right === 1 ? "" : "s"}` +
    ` · ${legWord("left")}: ${gTakes.left || "no"} take${gTakes.left === 1 ? "" : "s"}` +
    ` · center R:${gCtr.right ? "✓" : "—"} L:${gCtr.left ? "✓" : "—"}` +
    (r.outside_workspace ? " · ⚠ OUTSIDE WORKSPACE" : "") : "";
  // Process+save works with one leg (then ⇄ copy across), but says exactly what it will save
  const gr = !!gTakes.right, gl = !!gTakes.left;
  const fin = $("btn-rec-finish");
  fin.disabled = !(gr || gl);
  fin.textContent = !(gr || gl) ? "Process + save (record a leg first)"
    : (gr && gl) ? "Process + save (right + left)"
    : `Process + save (${gr ? "right" : "left"} only — copy across in step 3)`;

  // ---- workspace sweep ----
  const wSeg = r.segments || { right: 0, left: 0 };
  const activeSeg = wSeg[S.wsLeg] || 0;
  $("btn-wsrec-take").disabled = !inWs;
  $("btn-wsrec-undo").disabled = !inWs || r.active || !activeSeg;
  $("btn-wsrec-process").disabled = !activeSeg;
  $("btn-wsrec-take").textContent = (inWs && r.active) ? `■ stop pass (${r.n_samples})` : "▶ start pass";
  $("btn-wsrec-take").classList.toggle("active-rec", inWs && r.active);
  $("btn-wsrec-process").textContent =
    activeSeg ? `Process ${S.wsLeg} → build workspace` : "Process → build workspace";
  const swept = (S.wsAbdSweep[0] <= S.wsAbdSweep[1])
    ? ` · abd swept ${S.wsAbdSweep[0].toFixed(0)}…${S.wsAbdSweep[1].toFixed(0)}°` : "";
  $("wsrec-status").textContent = inWs ?
    `${S.wsLeg} leg · ${activeSeg} pass${activeSeg === 1 ? "" : "es"} recorded` + swept +
    (r.outside_workspace ? " · ⚠ outside current workspace" : "") : "";
}

/* ---------------- trajectory files ---------------- */
async function showTrajectory(name) {
  try {
    const d = await (await fetch(`/api/trajectory?name=${encodeURIComponent(name)}`)).json();
    if (d.error) { setBanner(d.error, "error", 6000); return; }
    S.traj = d; S.trajName = name;
    updateTrajLegBadges();
    trEd.view.render(); wsEd.view.render(); renderEE();
  } catch (e) { /* ignore */ }
}
$("btn-traj-show").onclick = () => showTrajectory($("traj-files").value);
$("btn-traj-export").onclick = () => {
  const name = $("traj-files").value;
  if (name) window.location = `/api/trajectory/export?name=${encodeURIComponent(name)}`;
};
$("traj-import").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  const r = await fetch("/api/trajectory/import", { method: "POST", body: fd });
  const d = await r.json();
  setBanner(d.ok ? "imported " + d.imported : d.error, d.ok ? "" : "error", 5000);
  e.target.value = "";
};

function updateFileLists(st) {
  fillSelect($("ws-files"), (st.workspace || {}).files || []);
  fillSelect($("traj-files"), st.trajectories || []);
  fillSelect($("pb-file"), st.trajectories || []);
  fillSelect($("del-file"), delFiles(st));
}
function fillSelect(sel, items) {
  const cur = sel.value;
  const want = items.join("|");
  if (sel.dataset.items === want) return;
  sel.dataset.items = want;
  sel.innerHTML = items.map((f) => `<option>${f}</option>`).join("");
  if (items.includes(cur)) sel.value = cur;
}

/* ---------------- file management (delete) ---------------- */
function delFiles(st) {
  return $("del-kind").value === "trajectory"
    ? (st.trajectories || [])
    : ((st.workspace || {}).files || []);
}
function hideDelConfirm() { $("del-confirm-row").classList.add("hidden"); }
$("del-kind").onchange = () => {
  hideDelConfirm();
  if (S.state) fillSelect($("del-file"), delFiles(S.state));   // swap the list to the new kind
};
$("del-file").onchange = hideDelConfirm;       // changing the target cancels a pending confirm
$("btn-del-file").onclick = () => {
  const name = $("del-file").value;
  if (!name) { setBanner("no file selected to delete", "warn", 2500); return; }
  const label = $("del-kind").value === "trajectory" ? "gait " : "workspace ";
  $("del-confirm-name").textContent = label + name;
  $("del-confirm-row").classList.remove("hidden");
};
$("btn-del-cancel").onclick = hideDelConfirm;
$("btn-del-confirm").onclick = async () => {
  const kind = $("del-kind").value, name = $("del-file").value;
  hideDelConfirm();
  if (!name) return;
  const url = kind === "trajectory" ? "/api/trajectory/delete" : "/api/workspace/delete";
  try {
    await api(url, { json: { name } });
    setBanner("deleted " + name, "", 3500);
  } catch (_) { /* api() already surfaced the error banner */ }
  if (S.state) fillSelect($("del-file"), delFiles(S.state));
};

/* ================================================================ digital twin */
// static/twin3d.js draws the homing-pose MJCF; this feeds it. Live motor angles by default, the
// gait preview while that runs (the same trajectory sample the playback would send). The angle
// map is qpos = sign * normalized deg: the drives are zeroed in the pose the MJCF was exported in,
// so there is no offset to fit, only a sign per motor (twinmap.py, persisted on the robot).
const TW = { view: null, signs: null, defaults: null, last: "" };

function setupTwin() {
  if (!window.Twin3D) return;
  TW.view = new Twin3D($("twin-canvas"));
  TW.view.load().then(() => { TW.last = ""; renderEE(); });
  for (const b of document.querySelectorAll("[data-twin-view]"))
    b.onclick = () => TW.view.view(b.dataset.twinView);
  fetch("/api/twin/map").then((r) => r.json()).then(applyTwinMap)
    .catch(() => twinBanner("could not load /api/twin/map — is the server up to date?"));
}

function applyTwinMap(d) {
  if (!d || !d.signs) return;
  TW.signs = d.signs;
  TW.defaults = d.defaults || d.signs;
  const box = $("twin-signs");
  if (!box.querySelector("select")) {
    box.insertAdjacentHTML("beforeend", MOTORS.map((n) =>
      `<label>${n} <select class="num small" data-twin-sign="${n}">` +
      `<option value="1">+1</option><option value="-1">−1</option></select></label>`).join("") +
      `<button id="btn-twin-defaults" class="btn small">defaults</button>`);
    for (const s of box.querySelectorAll("[data-twin-sign]"))
      s.onchange = () => saveTwinSigns({ [s.dataset.twinSign]: +s.value });
    $("btn-twin-defaults").onclick = () => saveTwinSigns(TW.defaults);
  }
  for (const s of box.querySelectorAll("[data-twin-sign]")) {
    const n = s.dataset.twinSign;
    s.value = String(TW.signs[n]);
    s.title = TW.signs[n] === TW.defaults[n] ? "default" : `changed from the default (${TW.defaults[n]})`;
    s.parentElement.classList.toggle("override", TW.signs[n] !== TW.defaults[n]);
  }
  TW.last = "";
  renderEE();
}

async function saveTwinSigns(signs) {
  try { applyTwinMap(await api("/api/twin/map", { json: { signs } })); }
  catch (_) { applyTwinMap({ signs: TW.signs, defaults: TW.defaults }); }   // revert the selects
}

/** Called by sensors.js on every IMU poll (~10 Hz): the filtered world-up in body axes, or null
 *  when the mount frame is not trustworthy (`frame`: "ok" | "uncal" | "conflict" | "none"). The twin's
 *  base turns about the torso origin by pitch and roll only; the world marker stays put. */
function twinSetAttitude(up, frame) {
  const tv = TW.view;
  if (!tv) return;
  const on = $("twin-imu").checked;
  const att = tv.setBase(on && frame === "ok" ? up : null);
  const el = $("twin-att");
  const sgn = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(1) + "°";
  if (!on) el.textContent = "base level (IMU off)";
  else if (frame === "ok")
    el.textContent = `base pitch ${sgn(att.pitch)} ${att.pitch >= 0 ? "nose down" : "nose up"} · ` +
      `roll ${sgn(att.roll)} ${att.roll >= 0 ? "right side down" : "left side down"}`;
  else el.textContent = frame === "conflict"
    ? "base level: the IMU axis tilts conflict — see Gyro calibration"
    : frame === "none" ? "base level: no IMU reading" : "base level: IMU mount not calibrated";
  el.classList.toggle("off", !on || frame !== "ok");
}
window.twinSetAttitude = twinSetAttitude;

function twinBanner(msg) {
  const b = $("twin-banner");
  b.textContent = msg || "";
  b.classList.toggle("hidden", !msg);
}

/** The pose the twin should show: {motor: normalized deg | null}, plus where it came from. */
function twinAngles() {
  const out = {};
  if (S.preview.on && S.traj) {
    for (const side of ["left", "right"]) {
      const tr = S.traj[side];
      if (!tr || !tr.path || !tr.path.length) continue;
      const [cam, thigh] = tr.path[previewIdx(tr, side, previewPhase())];
      Object.assign(out, { [side + ".abd"]: tr.abd_hold, [side + ".cam"]: cam, [side + ".thigh"]: thigh });
    }
    return { norm: out, src: "gait preview" };
  }
  const motors = (S.state && S.state.motors) || {};
  for (const n of MOTORS) {
    // telemetry (10 Hz) first; the 2 Hz state snapshot covers the first moments after page load
    const v = S.latest[n] ? S.latest[n].pos_norm : (motors[n] || {}).pos_norm;
    out[n] = Number.isFinite(v) ? v : null;
  }
  return { norm: out, src: "live" };
}

// kept under its old name: the workspace / trajectory / preview code calls renderEE() whenever
// something the leg view shows has changed
function renderEE() {
  const tv = TW.view;
  if (!tv) return;
  if (tv.error) { twinBanner(tv.error); return; }
  if (!tv.model || !TW.signs) return;
  const { norm, src } = twinAngles();
  const q = {};
  for (const n of MOTORS) q[n] = norm[n] === null || norm[n] === undefined ? null
    : TW.signs[n] * norm[n] * Math.PI / 180;
  const calOk = S.cal && S.cal.stage === "complete";
  const key = JSON.stringify([q, calOk, src]);
  if (key === TW.last) return;                 // nothing moved: skip the loop solve and the redraw
  TW.last = key;
  const loops = tv.setPose(q) || [];

  const deg = (v) => v === null || v === undefined ? "  —  " : ((v >= 0 ? "+" : "") + v.toFixed(1)).padStart(6);
  const lines = ["left", "right"].map((side) => {
    const L = loops.find((l) => /left/i.test(l.name) === (side === "left"));
    const r = L ? `  loop ${(L.resid * 1000).toFixed(2)} mm` : "";
    return `${side.padEnd(5)}  abd ${deg(norm[side + ".abd"])}°  cam ${deg(norm[side + ".cam"])}°  ` +
      `thigh ${deg(norm[side + ".thigh"])}°${r}`;
  });
  $("twin-status").textContent = `${src} (normalized deg)\n` + lines.join("\n");

  const silent = MOTORS.filter((n) => norm[n] === null);
  const open = loops.filter((l) => l.resid >= Twin3D.LOOP_TOL).map((l) => /left/i.test(l.name) ? "left" : "right");
  twinBanner(
    open.length ? `${open.join(" + ")} four-bar cannot close at these angles (gap shown below) — a sign ` +
      "or the zero is off. Showing the closest pose (leg orange)." :
    !calOk && src === "live" ? "NOT CALIBRATED — normalized angles mean nothing until the zero wizard " +
      "has run in the homing pose; the twin is drawn from them anyway." :
    silent.length && src === "live" ? `no reading from ${silent.join(", ")} — drawn at 0°` :
    src !== "live" ? "showing the GAIT PREVIEW, not the robot" : "");
}

/* ================================================================ playback */
// Show the cadence in Hz too: the slider is seconds/cycle, but a gait is thought about in Hz and
// the fast end is where that matters (0.2 s/cycle reads as nothing, 5 Hz reads as fast).
function updatePeriodLabel() {
  const p = +$("pb-period").value;
  $("pb-period-val").textContent = p.toFixed(p < 1 ? 2 : 1);
  $("pb-period-hz").textContent = `(${(1 / p).toFixed(p < 1 ? 1 : 2)} Hz)`;
}
$("pb-period").oninput = updatePeriodLabel;
updatePeriodLabel();
$("pb-leftphase").oninput = () => $("pb-leftphase-val").textContent = $("pb-leftphase").value;
$("pb-mode").onchange = () =>
  $("pb-current-params").style.display = $("pb-mode").value === "current" ? "" : "none";
$("pb-mode").onchange();

$("btn-pb-start").onclick = () => api("/api/playback/start", { json: {
  name: $("pb-file").value, legs: $("pb-legs").value, mode: $("pb-mode").value,
  period: +$("pb-period").value, left_phase: +$("pb-leftphase").value,
  current_limit: +$("pb-ilimit").value, kp: +$("pb-kp").value, ki: +$("pb-ki").value,
  kd: +$("pb-kd").value, ramp: +$("pb-ramp").value,
  max_track_err: +$("pb-trackerr").value,
  track_err_estop: $("pb-trackerr-estop").checked } });
$("btn-pb-stop").onclick = () => api("/api/playback/stop", { method: "POST" });

let pbPatchTimer = null;
function schedulePatch() {
  if (!(S.state && S.state.playback && S.state.playback.running)) return;
  if (pbPatchTimer) clearTimeout(pbPatchTimer);
  pbPatchTimer = setTimeout(() => api("/api/playback", { method: "PATCH", json: {
    period: +$("pb-period").value, left_phase: +$("pb-leftphase").value,
    max_track_err: +$("pb-trackerr").value,
    track_err_estop: $("pb-trackerr-estop").checked } }), 250);
}
$("pb-period").addEventListener("input", schedulePatch);
$("pb-leftphase").addEventListener("input", schedulePatch);
// live too: the whole point is retuning the limit while the gait runs, without restarting it
$("pb-trackerr").addEventListener("input", schedulePatch);
$("pb-trackerr-estop").addEventListener("change", schedulePatch);

function updatePlaybackUI(st) {
  const pb = st.playback;
  $("pb-phase").style.width = pb && pb.running ? (pb.phase * 100) + "%" : "0%";
  $("btn-pb-start").disabled = !!(pb && pb.running);

  // Peak tracking error, so "warn" is never "ignore": with the E-STOP decoupled this is the only
  // thing telling you how far behind the robot actually is.
  const el = $("pb-trackerr-state");
  if (!pb || pb.track_err_peak == null) { el.textContent = ""; el.className = "hint"; return; }
  const worst = pb.track_err_worst ? ` (${pb.track_err_worst})` : "";
  el.textContent = `peak ${pb.track_err_peak.toFixed(1)}°${worst}`;
  el.className = pb.track_err_over ? "hint warn-text" : "hint";
  if (pb.track_err_over && !pb.track_err_estop) {
    el.textContent += ` — OVER ${pb.max_track_err.toFixed(0)}°, E-STOP decoupled`;
  }
}

/* ================================================================ mock tools */
function mockSweep(side) {
  stopMockSweeps();
  let t = 0;
  S.mockTimers[side] = setInterval(() => {
    t += 0.1;
    const cam = 25 * Math.sin(t * 0.9);
    const thigh = 0.55 * cam + 10 * Math.cos(t * 0.45);
    const abd = 12 * Math.sin(t * 0.25);
    api("/api/mock/drag", { json: { motor: side + ".cam", norm_deg: cam } }).catch(() => {});
    api("/api/mock/drag", { json: { motor: side + ".thigh", norm_deg: thigh } }).catch(() => {});
    api("/api/mock/drag", { json: { motor: side + ".abd", norm_deg: abd } }).catch(() => {});
  }, 100);
}
function stopMockSweeps() {
  for (const k in S.mockTimers) clearInterval(S.mockTimers[k]);
  S.mockTimers = {};
  for (const side of ["left", "right"])
    for (const r of ROLES)
      api("/api/mock/drag", { json: { motor: side + "." + r, norm_deg: null } }).catch(() => {});
}
$("btn-mock-sweep-right").onclick = () => mockSweep("right");
$("btn-mock-sweep-left").onclick = () => mockSweep("left");
$("btn-mock-stop").onclick = stopMockSweeps;

/* ================================================================ boot */
function boot() {
  buildMotorCards();
  buildManualRows();
  wireGuards();
  setupWsCanvas();
  setupTrajCanvas();
  setupTwin();
  updateTrajLegBadges();
  refreshWorkspace().then(fitTrajView);
  pollState();
  setInterval(pollState, 500);
  setInterval(pollTelemetry, 100);
  setInterval(() => { if (!document.hidden) {
    drawCharts();
    updateManualLive();
    wsEd.view.render();
    trEd.view.render();
    drawAbd();
    renderEE();
  } }, 120);
}
boot();
