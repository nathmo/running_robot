/* Thermal identification panel. Loaded after app.js; reuses its globals (api, $, S, setBanner,
 * MOTORS). window.onThermalState(st) is called from applyState (~2 Hz).
 *
 * The experiment this drives: saturate ONE motor's torque for a fixed time, then read how far
 * its temperature rises. The operator is the instrument -- the drive's own sensor is on a node
 * nobody has identified -- so the panel has two jobs beyond pressing start.
 *
 * ONE: size the burst. A handheld probe resolves ~0.5 degC and the case rise is
 * E / (C_w + C_c), so the intuitive "12 A for 10 seconds" deposits 216 J and moves the case
 * 0.2 degC -- unreadable. The panel predicts both node rises before the run and refuses one that
 * would measure nothing, or that would put more than 60 degC into a winding no sensor watches.
 *
 * TWO: time the reading. The burst dumps its energy into the copper in seconds; a probe on the
 * CASE only sees it as that heat diffuses out, a minute or more later. Read too early and the
 * number is systematically low in a way that looks like a smaller heat input, so it biases the fit
 * instead of showing up as scatter. Hence the countdown, and hence "peak read at" being a field
 * rather than an assumption.
 */
"use strict";

const TH = {
  motor: "left.thigh",
  plan: null,                       // the read-only "would this wiggle run?" answer
  planInFlight: false, planAt: 0, planKey: null, zeroEpoch: null,
  amps: 20,        // 6 A x 5 s is 45 J -- it moves the case 0.04 degC and the gate refuses it.
  duration: 60,    // open on a burst that actually passes, not on a refusal.
  runs: [], cooldowns: [], summary: {}, limits: null, pred: null,
  pending: null,          // burst that has ended and is waiting to be saved
  cooldown: null,         // the cooldown curve currently being filled in
  peakSeen: 0,            // highest drive-reported temperature since the burst ended
};

function thInit() {
  $("th-motor").innerHTML = MOTORS.map((m) => `<option value="${m}">${m}</option>`).join("");
  $("th-motor").value = TH.motor;
  $("th-motor").onchange = (e) => { TH.motor = e.target.value; thRender(); thPredictSoon(); thIdentPlan(); };
  $("th-amps").oninput = (e) => { TH.amps = +e.target.value; thRender(); thPredictSoon(); };
  $("th-dur").oninput = (e) => {
    TH.duration = +e.target.value;
    $("th-dur-val").textContent = TH.duration.toFixed(0) + " s";
    thRender();
    thPredictSoon();
  };
  $("btn-th-suggest").onclick = thApplySuggestion;
  $("btn-th-start").onclick = thStart;
  $("btn-th-stop").onclick = () => api("/api/thermal/stop", { json: {} });
  $("btn-th-save").onclick = thSave;
  $("btn-th-cool-start").onclick = thCooldownStart;
  $("btn-th-cool-add").onclick = thCooldownAdd;
  $("th-confirm").onchange = thRender;
  $("th-rotor").onchange = () => {
    const free = $("th-rotor").value === "free";
    $("th-confirm-label").textContent = free
      ? "the motor is off the robot and NOTHING is attached to the shaft"
      : "the joint is clamped and cannot turn";
    $("th-confirm").checked = false;      // a different claim needs a fresh confirmation
    $("th-sine-row").classList.toggle("hidden", !free);
    thSineNote();
    thRender();
  };
  $("th-freq").oninput = thSineNote;
  $("th-amp").oninput = thSineNote;
  $("btn-th-ident").onclick = thIdentify;
  TH.identPlanFor = null;
  $("btn-th-ident-stop").onclick = () => api("/api/thermal/identify/stop", { json: {} });
  thRefresh();
  thPredict();
}

/* Predicted rise, refreshed whenever the knobs move. Debounced because it is a round trip and the
 * duration slider fires on every pixel. */
let thPredTimer = null;
function thPredictSoon() {
  if (thPredTimer) clearTimeout(thPredTimer);
  thPredTimer = setTimeout(thPredict, 200);
}
async function thPredict() {
  let d;
  try {
    d = await api("/api/thermal/predict", {
      json: { motor: TH.motor, amps: TH.amps, duration_s: TH.duration },
    });
  } catch (_) { return; }
  TH.pred = d;
  // `viable`, not `ok`: ok belongs to the response envelope and api() throws on ok:false, so a
  // verdict returned in that field would never reach this line at all.
  const p = d.prediction || {};
  const el = $("th-predict");
  const cal = d.calibrated ? "" :
    ` <span class="th-need">(from UNCALIBRATED placeholder parameters — treat as an order of ` +
    `magnitude, not a number)</span>`;
  el.className = "th-predict " + (d.viable ? "ok" : "bad");
  el.innerHTML =
    `<b>Predicted:</b> ${(p.energy_j || 0).toFixed(0)} J → ` +
    `case <b>+${(p.case_c || 0).toFixed(1)} °C</b> (what you will read), ` +
    `winding <b>+${(p.winding_c || 0).toFixed(0)} °C</b> (what nothing can see)${cal}` +
    (d.viable ? "" : `<div class="th-need">${d.why}</div>`);
  $("btn-th-start").disabled = !d.viable || !$("th-confirm").checked;
  const sg = d.suggestion || {};
  $("btn-th-suggest").textContent =
    `Use ${(sg.amps || 0).toFixed(0)} A × ${(sg.duration_s || 0).toFixed(0)} s`;
}
function thApplySuggestion() {
  const sg = (TH.pred && TH.pred.suggestion) || null;
  if (!sg) return;
  TH.amps = +sg.amps.toFixed(1);
  // clamp FIRST: the slider used to be clamped while TH.duration kept the unclamped value, so the
  // panel showed one duration and posted another.
  TH.duration = Math.min(Math.round(sg.duration_s), +$("th-dur").max);
  $("th-amps").value = TH.amps;
  $("th-dur").value = TH.duration;
  $("th-dur-val").textContent = TH.duration.toFixed(0) + " s";
  thPredict();
}

/* What the free-rotor sine asks of the mechanism, shown before anyone presses start. The
 * daemon re-validates the numbers; this is just so the operator sees the peak speed the
 * amplitude x frequency trade-off implies while turning the knobs. */
function thSineNote() {
  const f = +$("th-freq").value || 0;
  const a = +$("th-amp").value || 0;
  const peak = 2 * Math.PI * f * a;
  $("th-sine-note").textContent =
    `peak ${peak.toFixed(0)} °/s — the current comes from the rotor fighting its own inertia ` +
    `(∝ amplitude × freq²); the run stops early if it cannot draw the target current`;
}

async function thRefresh() {
  const d = await api("/api/thermal/runs");
  TH.runs = d.runs || [];
  TH.cooldowns = d.cooldowns || [];
  TH.summary = d.summary || {};
  TH.limits = d.limits || null;
  if (TH.limits) {
    $("th-dur").min = TH.limits.min_duration_s;
    $("th-dur").max = TH.limits.max_duration_s;
    $("th-amps").max = TH.limits.max_amps;
    if (TH.limits.free_freq_max_hz) {
      $("th-freq").min = TH.limits.free_freq_min_hz;
      $("th-freq").max = TH.limits.free_freq_max_hz;
      $("th-amp").max = TH.limits.free_amp_max_deg;
    }
  }
  thRender();
}

async function thStart() {
  if (!$("th-confirm").checked) {
    setBanner("confirm how the rotor is held before starting a burst", "warn", 5000);
    return;
  }
  TH.peakSeen = 0;
  const body = {
    motor: TH.motor, amps: TH.amps, duration_s: TH.duration,
    ambient_c: numOrNull("th-ambient"), rotor_mode: $("th-rotor").value,
  };
  if (body.rotor_mode === "free") {
    body.freq_hz = +$("th-freq").value;
    body.amp_deg = +$("th-amp").value;
  }
  try {
    await api("/api/thermal/start", { json: body });
  } catch (_) { /* the banner already carries the refusal */ }
}

async function thSave() {
  const body = {
    t_start_c: numOrNull("th-tstart"),
    t_peak_c: numOrNull("th-tpeak"),
    t_peak_at_s: numOrNull("th-tpeak-at"),
    ambient_c: numOrNull("th-ambient"),
    probe: $("th-probe").value,
    notes: $("th-notes").value || null,
  };
  await api("/api/thermal/save", { json: body });
  $("th-tpeak").value = "";
  $("th-tpeak-at").value = "";
  TH.pending = null;
  await thRefresh();
  setBanner("burst saved", "", 2500);
}

async function thCooldownStart() {
  const d = await api("/api/thermal/cooldown/start", {
    json: { motor: TH.motor, ambient_c: numOrNull("th-ambient"), probe: $("th-probe").value },
  });
  TH.cooldown = d.cooldown;
  await thRefresh();
}

async function thCooldownAdd() {
  if (!TH.cooldown) { setBanner("start a cooldown curve first", "warn", 3000); return; }
  const t = numOrNull("th-cool-t");
  const c = numOrNull("th-cool-temp");
  if (t === null || c === null) { setBanner("need both a time and a temperature", "warn", 3000); return; }
  const d = await api("/api/thermal/cooldown/point", { json: { id: TH.cooldown.id, t_s: t, temp_c: c } });
  TH.cooldown = d.cooldown;
  $("th-cool-temp").value = "";
  await thRefresh();
}

function numOrNull(id) {
  const v = $(id).value;
  return v === "" || v === null ? null : +v;
}


/* ------------------------------------------------------------------ joint identification
 * "Which motor is left.thigh?" is the question underneath every measurement here, and the panel
 * can answer it properly rather than by eye: the daemon holds the other five where they are and
 * reports how far EVERY joint moved, so a mis-mapped drive shows up as a number instead of as a
 * puzzled operator. Excursions come back in raw encoder degrees, so the verdict does not depend
 * on the calibration being right. */
function thIdentRunning() {
  return !!(S.state && S.state.identify && S.state.identify.running);
}

/* Ask whether the wiggle would run, and why not, BEFORE the operator presses anything.
 * Read-only: the endpoint queues nothing. This exists because both ways a wiggle can be refused
 * are invisible otherwise -- the workspace bound produced a message about a stop that was not
 * there, and a pre-move guard refusal landed after the HTTP call had already returned 200, so the
 * button simply did nothing. */
async function thIdentPlan() {
  // in-flight guard + a hard floor between requests: this endpoint is cheap to CALL and expensive
  // to SERVE, and it is served by the process running the control loop
  if (TH.planInFlight || performance.now() - TH.planAt < 2000) return;
  TH.planInFlight = true;
  TH.planAt = performance.now();
  const el = $("th-ident-note");
  let d;
  try {
    d = await api("/api/thermal/identify/plan", {
      json: { motor: TH.motor, amp_deg: 5.0, duration_s: 2.0 },
    });
  } catch (_) { return;
  } finally { TH.planInFlight = false; TH.planAt = performance.now(); }
  TH.plan = d;
  const p = d.plan || {};
  if (!d.viable) {
    el.className = "th-need";
    el.textContent = d.why;
  } else if (p.bound === "hard limits") {
    el.className = "hint";
    el.textContent = `will wiggle ±${p.amp_deg.toFixed(1)}° — ${p.note}`;
  } else {
    el.className = "hint";
    el.textContent = `will wiggle ±${p.amp_deg.toFixed(1)}° (bounded by the ${p.bound}) and ` +
      `report which joint actually moved`;
  }
  $("btn-th-ident").disabled = !d.viable || thIdentRunning();
}

async function thIdentify() {
  if (!$("th-confirm").checked) {
    setBanner("tick the confirmation below before moving anything", "warn", 5000);
    return;
  }
  try {
    await api("/api/thermal/identify", {
      json: { motor: TH.motor, amp_deg: 5.0, duration_s: 2.0, confirm_free: true },
    });
  } catch (_) { /* the banner already carries the refusal */ }
}

const TH_VERDICT = {
  confirmed: ["ok", "✓ confirmed"],
  mismatch: ["bad", "✗ WRONG JOINT"],
  coupled: ["warn", "⚠ more than one joint moved"],
  "no-motion": ["bad", "✗ nothing moved"],
  aborted: ["bad", "✗ aborted"],
};

function thRenderIdentify(i) {
  const el = $("th-ident");
  if (!i) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  if (i.running) {
    el.className = "th-ident active";
    el.innerHTML = i.holding
      ? `<b>${i.motor}</b> — holding before moving…`
      : `<b>${i.motor}</b> — wiggling ±${i.amp_deg.toFixed(1)}°, ` +
        `${Math.max(0, i.duration_s - i.elapsed_s).toFixed(1)} s left`;
    return;
  }
  const [cls, label] = TH_VERDICT[i.verdict] || ["", i.verdict];
  const exc = Object.entries(i.excursions || {}).sort((a, b) => b[1] - a[1]);
  el.className = "th-ident " + cls;
  el.innerHTML =
    `<b>${label}</b> — ${i.detail}` +
    `<div class="th-exc">` + exc.map(([n, v]) =>
      `<span class="th-pt${n === i.motor ? " sel" : ""}${v >= i.threshold_deg ? " moved" : ""}">` +
      `${n} ${v.toFixed(1)}°</span>`).join("") + `</div>` +
    `<div class="hint">selected joint boxed · moved = ≥ ${i.threshold_deg}° · ` +
    `worst tracking error ${i.track_err_deg.toFixed(1)}°</div>` +
    `<div class="row"><button class="btn tiny" id="btn-th-ident-clear">clear</button></div>`;
  const c = $("btn-th-ident-clear");
  if (c) c.onclick = () => api("/api/mode", { json: { mode: "LIMP" } });
}

/* Called from app.js applyState. `st.thermal` is the daemon's live burst state. */
function onThermalState(st) {
  // Re-ask the plan when something that changes the answer changes -- and NOT otherwise.
  //
  // The first version keyed on `st.calibration.zero_epoch` directly. `calibration` is only in the
  // COLD half of the state, so on every hot update the key read `undefined` and on every cold one
  // it read the epoch: the key flipped 4x a second, fired a plan request each time, and each
  // response ran applyState -> onThermalState -> flipped it back. A self-sustaining request storm
  // that ran _safe_room (~40 workspace evaluations) inside the process holding the 200 Hz CAN
  // loop, which is what made the whole UI feel stuck (2026-08-29).
  //
  // Two rules now: only ever LEARN a field that is present, and never let a response re-trigger
  // its own request.
  if (st.calibration && st.calibration.zero_epoch !== undefined) {
    TH.zeroEpoch = st.calibration.zero_epoch;
  }
  const key = `${TH.motor}/${TH.zeroEpoch}/${st.mode}`;
  const stale = performance.now() - TH.planAt > 15000;
  if (key !== TH.planKey || stale) { TH.planKey = key; thIdentPlan(); }
  thRenderIdentify(st.identify);
  const identing = !!(st.identify && st.identify.running);
  $("btn-th-ident").disabled = identing || !(TH.plan && TH.plan.viable);
  $("btn-th-ident-stop").disabled = !identing;
  const t = st.thermal;
  const live = $("th-live");
  const running = !!(t && t.running) || identing;
  $("btn-th-start").disabled = running || thIdentRunning() || !$("th-confirm").checked
    || !(TH.pred && TH.pred.viable);
  $("btn-th-stop").disabled = !running;

  if (!t) { live.classList.add("hidden"); return; }
  live.classList.remove("hidden");
  if (t.drive_t_peak_c != null) TH.peakSeen = Math.max(TH.peakSeen, t.drive_t_peak_c);

  if (running) {
    const left = Math.max(0, t.duration_s - t.elapsed_s);
    live.className = "th-live active";
    const head = t.free_rotor
      ? `<b>SINE RUNNING</b> — ${t.motor} tracking ±${t.sine_amp_deg.toFixed(0)}° at ` +
        `${t.freq_hz.toFixed(1)} Hz (sized for ${t.amps.toFixed(1)} A), ${left.toFixed(1)} s left`
      : `<b>BURST RUNNING</b> — ${t.motor} at ${t.amps.toFixed(1)} A, ${left.toFixed(1)} s left`;
    live.innerHTML = head +
      `<div class="hint">${t.free_rotor ? "measured " : t.reversals + " reversals · "}` +
      `${t.i_rms} A rms · ` +
      `travel ${t.travel_deg == null ? "—" : t.travel_deg.toFixed(1) + "°"} · ` +
      `peak ${t.peak_erpm.toFixed(0)} ERPM · drive ${t.drive_t_start_c}→${t.drive_t_peak_c} °C</div>`;
    return;
  }

  // finished: this is the WAIT, and the wait is part of the measurement
  const since = t.since_end_s || 0;
  TH.pending = t;
  const est = thPeakDelayHint();
  live.className = "th-live done" + (t.abort ? " bad" : "");
  live.innerHTML =
    (t.abort ? `<b>ABORTED:</b> ${t.abort}` : `<b>Burst complete</b> — ${t.motor}`) +
    `<div class="hint">${t.i_rms} A rms for ${t.duration_s.toFixed(0)} s · ` +
    `drive reported ${t.drive_t_start_c} → ${t.drive_t_peak_c} °C · ` +
    `${since.toFixed(0)} s since it ended</div>` +
    `<div class="th-wait">${est}</div>`;
}

function thPeakDelayHint() {
  const t = TH.pending;
  if (!t) return "";
  const since = t.since_end_s || 0;
  // Without a fitted model we cannot say exactly when the case peaks, so the panel gives the
  // honest version: watch the reading, record the turning point, and do not stop early. The
  // 60 s floor is a lower bound from the winding time constants this class of motor has.
  if (since < 60) {
    return `⏳ <b>Do not read the peak yet.</b> The copper is hot but the case has barely felt it. ` +
      `Keep watching the probe — the peak is typically 1–3 min after the burst — and record the ` +
      `turning point, not the first number you see. (${(60 - since).toFixed(0)} s until the ` +
      `earliest plausible peak.)`;
  }
  return `Watch for the turning point, then enter the peak and how many seconds after the burst ` +
    `you read it. Save even if you are unsure — an unannotated run can be filled in later.`;
}

function thRender() {
  const s = TH.summary[TH.motor] || null;
  const el = $("th-progress");
  if (!s) {
    el.innerHTML = `<span class="hint">No runs recorded for ${TH.motor} yet.</span>`;
  } else {
    const needs = (s.needs || []);
    el.innerHTML =
      `<div class="th-count">${s.usable}/${s.bursts} annotated burst(s) at ` +
      `${[...new Set(s.durations)].sort((a, b) => a - b).join(", ") || "—"} s · ` +
      `${s.cooldown_points} cooldown point(s) over ${(s.cooldown_span_s / 60).toFixed(0)} min</div>` +
      (needs.length
        ? `<div class="th-need">Still needed: ${needs.join("; ")}</div>`
        : `<div class="th-ready">✓ enough data to fit this motor</div>`);
  }
  $("th-runs").innerHTML = TH.runs.filter((r) => r.motor === TH.motor).slice(-12).reverse()
    .map((r) => {
      const rise = (r.t_start_c != null && r.t_peak_c != null)
        ? `${r.t_start_c}→${r.t_peak_c} °C (+${(r.t_peak_c - r.t_start_c).toFixed(1)})`
        : `<span class="th-need">not annotated</span>`;
      return `<tr><td>${r.created_str.slice(11)}</td>` +
        `<td>${r.envelope.duration_s.toFixed(0)} s</td>` +
        `<td>${r.summary.i_rms.toFixed(1)} A</td>` +
        `<td>${rise}</td>` +
        `<td>${r.drive_t_start_c}→${r.drive_t_peak_c}</td>` +
        `<td>${r.aborted ? "⚠" : ""}<button class="btn tiny" data-del="${r.id}">✕</button></td></tr>`;
    }).join("") || `<tr><td colspan="6" class="hint">no bursts yet</td></tr>`;
  $("th-runs").querySelectorAll("[data-del]").forEach((b) => {
    b.onclick = async () => { await api("/api/thermal/delete", { json: { id: b.dataset.del } }); thRefresh(); };
  });

  const c = TH.cooldown;
  $("th-cool-body").innerHTML = !c ? `<span class="hint">no cooldown curve open</span>`
    : `<div class="hint">${c.motor} · ${c.points.length} point(s)</div>` +
      `<div class="th-points">` + c.points.map(([t, v], i) =>
        `<span class="th-pt">${(t / 60).toFixed(1)}min ${v}°C` +
        `<button class="btn tiny" data-dropi="${i}">✕</button></span>`).join("") + `</div>`;
  $("th-cool-body").querySelectorAll("[data-dropi]").forEach((b) => {
    b.onclick = async () => {
      const d = await api("/api/thermal/cooldown/drop",
        { json: { id: TH.cooldown.id, index: +b.dataset.dropi } });
      TH.cooldown = d.cooldown; thRefresh();
    };
  });
}

window.onThermalState = onThermalState;
document.addEventListener("DOMContentLoaded", thInit);
