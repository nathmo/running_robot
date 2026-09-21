/* Sense HAT (B) panel. Loaded after app.js, reuses its globals (api, $, fmt, StripChart,
 * setBanner, Inertia3D). Polls /api/sensors on its own (~10 Hz) rather than riding /api/state: the
 * IMU is a fast signal and the state poll is 2 Hz.
 *
 * Values are in ROBOT BODY axes (X forward, Y left, Z up) once the mount calibration below has been
 * run, and in the IMU's raw chip axes before that — the panel says which. Roll/pitch are
 * gravity-referenced; yaw is gyro-integrated and drifts. */
"use strict";

const SNS = { seq: 0, charts: null, rpy: [0, 0, 0], down: false,
  mount: null, mountKey: "", viewer: null, meshBuf: undefined, upBody: [0, 0, 1],
  seqActive: false, noising: false, noiseKey: "" };

const TILT_ORDER = ["fwd", "left", "right", "back"];
const TILT_NAME = { fwd: "forward", left: "left", right: "right", back: "backward" };

/* key, label, decimals — one row of a readout group */
const SNS_GROUPS = {
  "sns-acc": [["ax", "X", 3], ["ay", "Y", 3], ["az", "Z", 3], ["acc_mag", "|a|", 3]],
  "sns-gyr": [["gx", "X", 2], ["gy", "Y", 2], ["gz", "Z", 2], ["gyro_mag", "|ω|", 2]],
  "sns-mag": [["mx", "X", 1], ["my", "Y", 1], ["mz", "Z", 1], ["heading", "heading °", 1]],
  "sns-air": [["temp", "air °C", 2], ["humidity", "RH %", 1], ["pressure", "hPa", 2],
              ["temp_baro", "baro °C", 1], ["temp_imu", "IMU die °C", 1]],
  "sns-light": [["lux", "lux", 1], ["cct", "colour K", 0], ["clear", "clear", 0]],
  "sns-adc": [["adc0", "AIN0", 3], ["adc1", "AIN1", 3], ["adc2", "AIN2", 3], ["adc3", "AIN3", 3]],
};

function buildSensorRows() {
  for (const [id, rows] of Object.entries(SNS_GROUPS)) {
    $(id).innerHTML = rows.map(([k, label]) =>
      `<span>${label}</span><b id="sv-${k}">—</b>`).join("");
  }
  SNS.charts = {
    att: new StripChart($("sns-c-att"), COLORS.pos, 60, COLORS.target),   // pitch + roll
    gyr: new StripChart($("sns-c-gyr"), COLORS.cur, 60),
    acc: new StripChart($("sns-c-acc"), COLORS.good, 60),
    temp: new StripChart($("sns-c-temp"), COLORS.temp, 60),
  };
}

/* ================================================================ polling */
async function pollSensors() {
  let d;
  try { d = await (await fetch(`/api/sensors?since=${SNS.seq}`)).json(); } catch (e) { return; }

  const down = !d.available;
  $("sns-down").classList.toggle("hidden", !down);
  $("sns-body").classList.toggle("hidden", down);
  if (down) {
    $("sns-down").textContent = "Sense HAT (B) not reading: " + (d.error || "unknown reason");
    $("sns-status").textContent = "";
    if (window.twinSetAttitude) twinSetAttitude(null, "none");
    return;
  }

  SNS.seq = d.seq;
  const t = d.t || [], s = d.series || {};
  for (let i = 0; i < t.length; i++) {
    SNS.charts.att.push(t[i], s.pitch[i], s.roll[i]);
    SNS.charts.gyr.push(t[i], s.gyro_mag[i]);
    SNS.charts.acc.push(t[i], s.acc_mag[i]);
    SNS.charts.temp.push(t[i], s.temp[i]);
  }
  if (t.length) SNS.lastT = t[t.length - 1];

  const v = d.values || {};
  for (const rows of Object.values(SNS_GROUPS))
    for (const [k, , dec] of rows) {
      const el = $("sv-" + k);
      if (el) { el.textContent = fmt(v[k], dec); el.classList.toggle("stale", v[k] === null || v[k] === undefined); }
    }
  $("sns-roll").textContent = fmt(v.roll);
  $("sns-pitch").textContent = fmt(v.pitch);
  $("sns-yaw").textContent = fmt(v.yaw);
  SNS.rpy = [v.roll || 0, v.pitch || 0, v.yaw || 0];

  // colour swatch: raw RGBC counts normalised against the strongest channel (the sensor measures
  // relative channel response, not an sRGB colour)
  const rgb = v.rgb;
  if (rgb && rgb.every((x) => x !== null)) {
    const m = Math.max(1, ...rgb);
    $("sns-swatch").style.background =
      `rgb(${rgb.map((x) => Math.round(255 * x / m)).join(",")})`;
  }

  const chips = (d.chips || []).join(", ");
  const cal = (d.mount || {}).calibrated;
  $("sns-status").textContent =
    `${chips} — ${cal ? "body axes" : "RAW CHIP AXES (mount not calibrated)"}` +
    `${d.mag_live ? "" : ", mag idle"}${d.i2c_errors ? ` — ${d.i2c_errors} read errors` : ""}`;
  const bs = d.bias_status || {};
  $("sns-bias").textContent = "gyro zero: " + (bs.msg || "—");
  $("sns-bias").className = "hint" + (bs.state === "moving" ? " warn-text" : "");
  $("sns-mount").textContent = mountHint(d);
  SNS.upBody = [v.ax || 0, v.ay || 0, v.az || 0];
  updateMountUI(d);
  updateNoiseUI(d);
  // the digital twin tilts its base by the filtered attitude (app.js) — only in body axes
  const m = d.mount || {};
  const frame = !m.calibrated ? "uncal" : (m.check || {}).conflict ? "conflict" : "ok";
  if (window.twinSetAttitude) twinSetAttitude(frame === "ok" ? d.up_body : null, frame);
}

/** One line on where the frame stands. Before calibration it reports which CHIP axis gravity sits
 *  on (the cheap sanity check that the HAT is where we think); after, it reports the residual tilt
 *  of the reference pose, which should be ~0 by construction. */
function mountHint(d) {
  const raw = d.acc_chip || [];
  const cal = (d.mount || {}).calibrated;
  if (raw.some((x) => x === null || x === undefined)) return "";
  const mag = Math.hypot(...raw);
  if (Math.abs(mag - 1) > 0.08) return "in motion — frame checks need the robot at rest";
  if (!cal) {
    const i = raw.map(Math.abs).indexOf(Math.max(...raw.map(Math.abs)));
    return `chip axes: gravity on ${raw[i] < 0 ? "−" : "+"}${"XYZ"[i]} — run the mount calibration below`;
  }
  const v = d.values || {};
  return `body axes: ${fmt(Math.hypot(v.roll || 0, v.pitch || 0), 1)}° off the reference upright`;
}

/* ================================================================ mount calibration */
const deg1 = (x) => (x === null || x === undefined ? "—" : `${x.toFixed(1)}°`);
const axisName = (a) => (a ? `chip ${a[0][0] === "-" ? "−" : "+"}${a[0][1].toUpperCase()}` : "—");

function updateMountUI(d) {
  const m = d.mount || {};
  SNS.mount = m;
  const cs = d.capture_status || {};
  const ss = d.seq_status || {};
  const chk = m.check || {};

  const lvl = m.captures && m.captures.level;
  $("cap-level-status").textContent = (cs.kind === "level" ? cs.msg : "") ||
    (lvl ? `captured (|a| ${fmt(lvl.acc_mag_g, 3)} g)` : "not captured");
  $("cap-level-status").className = "hint" +
    (cs.kind === "level" && ["moving", "tilt"].includes(cs.state) ? " warn-text" : "");

  // ---- the sequence: button, live tilt, the big instruction line
  SNS.seqActive = !!ss.active;
  $("btn-seq").textContent = ss.active ? "✕ Cancel sequence" : "▶ Start tilt sequence";
  $("btn-seq").classList.toggle("primary", !ss.active);
  $("seq-live").textContent = ss.active && Number.isFinite(ss.tilt)
    ? `tilt from upright ${ss.tilt.toFixed(1)}°` : "";
  // the instruction line stays up after the run ends (done / cancelled), but only in a page that
  // started or watched it — a fresh page load does not replay an old "complete"
  const showMsg = ss.active || (SNS.seqShown && ["done", "cancelled"].includes(ss.state));
  $("seq-msg").classList.toggle("hidden", !showMsg);
  if (showMsg) {
    const step = ss.active ? `${ss.i + 1}/${ss.order.length} — ` : "";
    $("seq-msg").textContent = step + ss.msg;
    $("seq-msg").className = "seq-msg" + (ss.state === "retry" ? " retry" : ss.state === "done" ? " done" : "");
  }
  if (ss.active) SNS.seqShown = true;

  // ---- one chip per tilt: captured angle, weak / old flags, fit residual, redo
  const tilts = m.tilts || {}, res = chk.residual_deg || {};
  const html = TILT_ORDER.map((k) => {
    const t = tilts[k];
    const now = ss.active && ss.which === k;
    let cls = "seq-tilt", txt;
    if (now) {
      cls += " now";
      txt = ss.phase === "capturing" ? "capturing…" : ss.phase === "tilt" ? "tilt now" : "next";
    } else if (!t) txt = "—";
    else if (t.legacy) { cls += " weak"; txt = "old capture"; }
    else {
      const r = res[k];
      cls += t.weak || (r !== undefined && r > 15) ? " weak" : " ok";
      txt = `${t.tilt_deg.toFixed(1)}°` + (r !== undefined ? ` · fit ${r.toFixed(1)}°` : "");
    }
    return `<div class="${cls}" title="${t && t.legacy ? t.legacy : ""}"><span>${TILT_NAME[k]}<br>` +
      `<span class="hint">${txt}</span></span>` +
      (ss.active ? "" : `<button class="btn small" data-redo="${k}" title="redo this tilt only">↻</button>`) +
      `</div>`;
  }).join("");
  if (html !== SNS.tiltHtml) { SNS.tiltHtml = html; $("seq-tilts").innerHTML = html; }

  // ---- the result, and whether its parts agree
  $("seq-axes").innerHTML = m.calibrated && m.axes
    ? `forward = <b>${axisName(m.axes.fwd)}</b> (${deg1(m.axes.fwd[1])} off) · ` +
      `left = <b>${axisName(m.axes.left)}</b> (${deg1(m.axes.left[1])} off) · ` +
      `up = <b>${axisName(m.axes.up)}</b>`
    : `<span class="warn-text">axes not set — pitch and roll are not yet distinguishable</span>`;
  const lines = [];
  const pair = (p, a, b) => {
    const x = (chk.pairs || {})[p];
    if (!x) return;
    if (x.n < 2) { lines.push(`${a} only — do ${b} too to check it`); return; }
    if (x.excluded) lines.push(`<span class="warn-text">✗ ${a} and ${b} point the SAME way ` +
      `(${deg1(x.disagree_deg)} apart) — one was tilted the wrong way; redo them. Left out of ` +
      `the fit.</span>`);
    else lines.push(`${a} ↔ ${b} agree within ${deg1(x.disagree_deg)}` +
      (x.disagree_deg > 15 ? ` <span class="warn-text">— sloppy, redo them</span>` : " ✓"));
  };
  pair("fore_aft", "forward", "backward");
  pair("lateral", "left", "right");
  if (chk.conflict)
    lines.push(`<span class="warn-text">✗ fore/aft and left/right describe a MIRROR ` +
      `(${deg1(chk.right_angle_deg)} off): one pair is the wrong way round. Using fore/aft alone ` +
      `until one of them is flipped — ⚖ Balance is refused meanwhile.</span>`);
  else if (Number.isFinite(chk.right_angle_deg))
    lines.push(`fore/aft ⟂ left/right within ${deg1(chk.right_angle_deg)}` +
      (chk.right_angle_deg > 15 ? ` <span class="warn-text">— check the tilts</span>` : " ✓"));
  if (tilts.fwd && tilts.fwd.legacy)
    lines.push(`<span class="warn-text">axes still from the ${tilts.fwd.legacy} — run the ` +
      `sequence to replace it</span>`);
  $("seq-checks").innerHTML = lines.join("<br>");

  const flip = m.flip || {};
  $("btn-flip-fa").classList.toggle("on", !!flip.fore_aft);
  $("btn-flip-lat").classList.toggle("on", !!flip.lateral);
  $("btn-flip-fa").textContent = "⇄ flip fore/aft" + (flip.fore_aft ? " (flipped)" : "");
  $("btn-flip-lat").textContent = "⇄ flip left/right" + (flip.lateral ? " (flipped)" : "");

  const key = JSON.stringify([m.R_chip_to_body, m.calibrated]);
  if (key !== SNS.mountKey) { SNS.mountKey = key; refreshFrameView(); }
}

/* ================================================================ noise recorder */
function updateNoiseUI(d) {
  const ns = d.noise_status || {};
  SNS.noising = ns.state === "recording";
  $("btn-noise").textContent = SNS.noising ? "■ Stop & analyse" : "● Record";
  $("btn-noise").classList.toggle("active-rec", SNS.noising);
  $("noise-status").textContent = SNS.noising
    ? `recording ${fmt(ns.seconds, 1)} s — do not touch the robot (20 s or more is a good record)`
    : ns.msg || "";
  $("noise-status").className = "hint" + (["moving", "error"].includes(ns.state) ? " warn-text" : "");

  const r = d.noise_result;
  const key = r ? `${r.file}|${r.n}` : "";
  if (key === SNS.noiseKey) return;
  SNS.noiseKey = key;
  if (!r || !r.ok) { $("noise-result").innerHTML = ""; return; }
  const f = (v, k) => (v === null || v === undefined ? "—" : v.toFixed(k));
  const row = (label, arr, k) => `<tr><td>${label}</td>` +
    [0, 1, 2].map((i) => `<td>${arr ? f(arr[i], k) : "—"}</td>`).join("") + `</tr>`;
  const ax = r.frame === "body" ? ["X fwd", "Y left", "Z up"] : ["chip X", "chip Y", "chip Z"];
  const att = (a) => (a ? `${f(a.rms, 3)}° RMS (${f(a.pp, 2)}° p-p)` : "—");
  $("noise-result").innerHTML =
    `<table><tr><th></th>${ax.map((a) => `<th>${a}</th>`).join("")}</tr>` +
    row("accel noise, mg RMS", r.acc_rms_mg, 2) +
    row("accel density, µg/√Hz", r.acc_density_ug, 0) +
    row("gyro noise, °/s RMS", r.gyr_rms_dps, 3) +
    row("gyro density, °/s/√Hz", r.gyr_density_dps, 4) +
    row("gyro left after the zero, °/s", r.gyr_mean_dps, 3) +
    row("gyro 1 s wander, °/s", r.gyr_wander_dps, 4) +
    `</table>` +
    `<div class="verdict">pitch ${att(r.pitch)} · roll ${att(r.roll)} — what ⚖ Balance steers on</div>` +
    `<div class="verdict ${r.still ? "" : "bad"}">${r.still ? "✓ " : "⚠ "}${r.still_why}</div>` +
    `<div class="hint">${fmt(r.seconds, 1)} s, ${r.n} samples at ${fmt(r.rate_hz, 1)} Hz ` +
    `(jitter ${fmt(r.jitter_ms, 2)} ms, max gap ${fmt(r.max_gap_ms, 1)} ms), |a| ${fmt(r.acc_mag_g, 4)} g` +
    (r.file ? ` — saved data/imu_noise/${r.file}` : r.save_error ? ` — NOT saved: ${r.save_error}` : "") +
    `</div>`;
}

/* ================================================================ 3D frame view */
const AX_COLORS = [[0.90, 0.28, 0.28], [0.30, 0.80, 0.36], [0.35, 0.62, 1.0]];   // X, Y, Z

async function ensureFrameViewer() {
  if (!SNS.viewer && window.Inertia3D) SNS.viewer = new Inertia3D($("sns-frame-canvas"));
  if (SNS.meshBuf === undefined) {
    SNS.meshBuf = null;
    try { SNS.meshBuf = await (await fetch("/api/mesh/bodyNCS-v1.stl")).arrayBuffer(); } catch (e) { /* no mesh */ }
  }
  return SNS.viewer;
}

/** Mesh side of the frame view. Parsing the STL is expensive, so this runs only when the mount
 *  changes or the toggle flips — never on the live tick. */
async function refreshFrameView() {
  const v = await ensureFrameViewer();
  if (!v || !v.ok) return;
  if ($("frame-showmesh").checked && SNS.meshBuf) v.setMesh(SNS.meshBuf, 0.001);
  else v.clearMesh();
  drawFrameSegments();
}

/** Draw the base frame and the IMU chip's own axes (as the mount calibration places them), both
 *  in the base body frame at the origin — where the HAT physically sits is not modelled. Cheap
 *  enough to re-run on the live tick so the measured up-vector animates. */
function drawFrameSegments() {
  const v = SNS.viewer;
  if (!v || !v.ok) return;
  const m = SNS.mount || {};

  const L = 0.12;                     // body-frame triad arm length, metres
  const segs = [];
  for (let i = 0; i < 3; i++) {
    const e = [0, 0, 0]; e[i] = L;
    segs.push({ a: [0, 0, 0], b: e, color: AX_COLORS[i] });
  }

  const r = [0, 0, 0];
  {
    // The chip axes expressed in body coordinates are the COLUMNS of R_chip_to_body.
    const R = m.R_chip_to_body || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let c = 0; c < 3; c++) {
      const dir = [R[0][c], R[1][c], R[2][c]];
      segs.push({ a: r, b: r.map((x, k) => x + dir[k] * L * 0.6),
        color: AX_COLORS[c].map((x) => Math.min(1, x * 0.75 + 0.25)) });
    }
    if ($("frame-showlive").checked) {
      const u = SNS.upBody, n = Math.hypot(...u) || 1;
      segs.push({ a: r, b: r.map((x, k) => x + (u[k] / n) * L), color: [1.0, 0.82, 0.25] });
    }
  }
  v.setSegments(segs);

  const key = (c, t) => `<span class="sns-axis-key"><i style="background:rgb(${c.map((x) => x * 255 | 0)})"></i>${t}</span>`;
  $("sns-frame-legend").innerHTML =
    key(AX_COLORS[0], "X fwd") + key(AX_COLORS[1], "Y left") + key(AX_COLORS[2], "Z up") +
    key([0.9, 0.9, 0.9], "chip axes (short)") + key([1, 0.82, 0.25], "measured up");
}

/* ================================================================ artificial horizon */
function drawHorizon() {
  const c = $("sns-horizon"), g = c.getContext("2d");
  const w = c.width, h = c.height, cx = w / 2, cy = h / 2, R = Math.min(w, h) / 2 - 8;
  const [roll, pitch] = SNS.rpy;
  const PPD = R / 55;                       // pixels per degree of pitch on the ladder

  g.clearRect(0, 0, w, h);
  g.save();
  g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.clip();
  g.translate(cx, cy);
  g.rotate(-roll * Math.PI / 180);
  g.translate(0, pitch * PPD);

  g.fillStyle = "#2b4a6b"; g.fillRect(-R * 2, -R * 3, R * 4, R * 3);          // sky
  g.fillStyle = "#4a3a26"; g.fillRect(-R * 2, 0, R * 4, R * 3);               // ground
  g.strokeStyle = "#d7dee8"; g.lineWidth = 2;
  g.beginPath(); g.moveTo(-R * 2, 0); g.lineTo(R * 2, 0); g.stroke();         // horizon

  g.lineWidth = 1; g.font = "10px ui-monospace, monospace";
  g.fillStyle = "#d7dee8"; g.textAlign = "center";
  for (let p = -40; p <= 40; p += 10) {
    if (!p) continue;
    const y = p * PPD, half = R * 0.28;
    g.beginPath(); g.moveTo(-half, y); g.lineTo(half, y); g.stroke();
    g.fillText(String(-p), 0, y - 3);
  }
  g.restore();

  // fixed aircraft symbol + roll pointer, drawn in screen space
  g.strokeStyle = "#ffd23f"; g.lineWidth = 2.5;
  g.beginPath();
  g.moveTo(cx - R * 0.45, cy); g.lineTo(cx - R * 0.14, cy);
  g.moveTo(cx + R * 0.14, cy); g.lineTo(cx + R * 0.45, cy);
  g.moveTo(cx, cy - 3); g.lineTo(cx, cy + 3);
  g.stroke();

  g.strokeStyle = "#8b97a8"; g.lineWidth = 1;
  g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();
  const a = (-roll - 90) * Math.PI / 180;    // roll index sliding around the top of the dial
  g.fillStyle = "#ffd23f";
  g.beginPath();
  g.moveTo(cx + R * Math.cos(a), cy + R * Math.sin(a));
  g.lineTo(cx + (R - 9) * Math.cos(a - 0.06), cy + (R - 9) * Math.sin(a - 0.06));
  g.lineTo(cx + (R - 9) * Math.cos(a + 0.06), cy + (R - 9) * Math.sin(a + 0.06));
  g.closePath(); g.fill();
}

function drawSensorCharts() {
  if (!SNS.charts || document.hidden) return;
  const now = SNS.lastT || 0;
  SNS.charts.att.draw(now);
  SNS.charts.gyr.draw(now);
  SNS.charts.acc.draw(now);
  SNS.charts.temp.draw(now);
  drawHorizon();
  if ($("frame-showlive").checked) drawFrameSegments();
}

/* ================================================================ actions + boot */
async function capture(kind, btn, msg) {
  const b = $(btn);
  b.disabled = true;
  try {
    await api("/api/sensors/capture", { json: { kind } });
    setBanner(msg, "", 2500);
  } catch (e) { /* banner already set by api() */ } finally {
    setTimeout(() => { b.disabled = false; }, 2000);
  }
}

$("btn-gyro-bias").onclick = () =>
  capture("gyro", "btn-gyro-bias", "Averaging the gyro zero — hold the robot still…");
$("btn-cap-level").onclick = () =>
  capture("level", "btn-cap-level", "Capturing the upright reference — hold the robot still…");

$("btn-seq").onclick = async () => {
  try {
    if (SNS.seqActive) await api("/api/sensors/sequence", { json: { action: "cancel" } });
    else {
      SNS.seqShown = true;
      await api("/api/sensors/sequence", { json: { action: "start" } });
    }
  } catch (e) { /* banner already set by api() */ }
};
$("seq-tilts").onclick = (e) => {
  const b = e.target.closest("[data-redo]");
  if (!b) return;
  SNS.seqShown = true;
  api("/api/sensors/sequence", { json: { action: "redo", tilt: b.dataset.redo } }).catch(() => {});
};
const flipPair = (pair) => {
  const on = !((SNS.mount || {}).flip || {})[pair];
  api("/api/sensors/mount", { json: { flip: { [pair]: on } } }).catch(() => {});
};
$("btn-flip-fa").onclick = () => flipPair("fore_aft");
$("btn-flip-lat").onclick = () => flipPair("lateral");

$("btn-noise").onclick = async () => {
  try {
    if (SNS.noising) await api("/api/sensors/noise", { json: { action: "stop" } });
    else {
      await api("/api/sensors/noise", { json: { action: "start" } });
      setBanner("Recording IMU noise — leave the robot completely still", "", 3000);
    }
  } catch (e) { /* banner already set by api() */ }
};

$("btn-mount-reset").onclick = async () => {
  if (!confirm("Forget the upright reference and the axes? Values go back to chip axes, and " +
      "⚖ Balance is refused until they are redone.")) return;
  try {
    await api("/api/sensors/mount/reset", { json: {} });
    setBanner("Mount calibration reset — values are back in chip axes", "warn", 4000);
  } catch (e) { /* banner already set by api() */ }
};

$("frame-showmesh").onchange = refreshFrameView;
$("frame-showlive").onchange = refreshFrameView;

for (const b of document.querySelectorAll("[data-mock-pose]"))
  b.onclick = () => api("/api/mock/sensors", { json: { pose: b.dataset.mockPose } }).catch(() => {});

buildSensorRows();
pollSensors();
setInterval(pollSensors, 100);
setInterval(drawSensorCharts, 120);
