/* Sense HAT (B) panel. Loaded after app.js, reuses its globals (api, $, fmt, StripChart,
 * setBanner, Inertia3D). Polls /api/sensors on its own (~10 Hz) rather than riding /api/state: the
 * IMU is a fast signal and the state poll is 2 Hz.
 *
 * Values are in ROBOT BODY axes (X forward, Y left, Z up) once the mount calibration below has been
 * run, and in the IMU's raw chip axes before that — the panel says which. Roll/pitch are
 * gravity-referenced; yaw is gyro-integrated and drifts. */
"use strict";

const SNS = { seq: 0, charts: null, rpy: [0, 0, 0], down: false,
  mount: null, mountKey: "", viewer: null, meshBuf: undefined, levering: false,
  levEdited: false, upBody: [0, 0, 1] };

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
function updateMountUI(d) {
  const m = d.mount || {};
  SNS.mount = m;
  const cs = d.capture_status || {};
  const ls = d.lever_status || {};

  const capMsg = (kind) => (cs.kind === kind ? cs.msg : "");
  const capCls = (kind) => "hint" + (cs.kind === kind && ["moving", "tilt", "weak"].includes(cs.state) ? " warn-text" : "");
  const lvl = m.captures && m.captures.level;
  $("cap-level-status").textContent = capMsg("level") ||
    (lvl ? `captured (|a| ${fmt(lvl.acc_mag_g, 3)} g)` : "not captured");
  $("cap-level-status").className = capCls("level");
  const fwd = m.captures && m.captures.forward;
  $("cap-fwd-status").textContent = capMsg("forward") ||
    (fwd ? `tilt capture: ${fmt(fwd.tilt_deg, 1)}° nose-down` +
           (fwd.weak ? ` — SHALLOW: 1° of roll while tipping = ~${fmt(fwd.roll_sensitivity_deg, 0)}° of fore-aft error; redo at 10–20°` : "")
         : "no tilt capture");
  $("cap-fwd-status").className = "hint" + ((cs.kind === "forward" && ["moving","tilt","weak"].includes(cs.state)) || (fwd && fwd.weak) ? " warn-text" : "");

  const cc = m.cross_check_deg;
  if (cc === null || cc === undefined) {
    $("sns-crosscheck").textContent = m.fwd_chip || m.fwd_declared ? "" :
      "fore-aft axis not set — pitch and roll are not yet distinguishable";
    $("sns-crosscheck").className = "hint" + (m.fwd_chip || m.fwd_declared ? "" : " warn-text");
  } else {
    $("sns-crosscheck").textContent =
      `measured vs declared forward: ${fmt(cc, 1)}° apart` +
      (cc > 15 ? " — the HAT is not bolted on square (the measured axis is the one in use)" : " ✓");
    $("sns-crosscheck").className = "hint" + (cc > 15 ? " warn-text" : "");
  }

  // selects/inputs follow the server unless the user is mid-edit
  const sel = $("sns-fwd-axis");
  if (document.activeElement !== sel) sel.value = m.fwd_declared || "";
  const use = $("sns-lever-use");
  if (document.activeElement !== use) use.value = m.lever_use || "cad";
  if (!SNS.levEdited && m.lever_cad)
    ["x", "y", "z"].forEach((k, i) => { $("lev-" + k).value = m.lever_cad[i]; });

  $("lever-fit-status").textContent = ls.msg ? `${ls.msg}${ls.state === "recording" ? ` (${ls.n} samples)` : ""}` : "";
  $("lever-fit-status").className = "hint" + (ls.state === "error" ? " warn-text" : "");
  $("btn-lever-fit").textContent = ls.state === "recording" ? "⏹ Stop & fit" : "⟳ Fit by rocking…";
  SNS.levering = ls.state === "recording";

  const f = m.lever_fit;
  const bits = [];
  if (f && f.ok) {
    bits.push(`fit [${f.r.map((x) => x.toFixed(3)).join(", ")}] m about ${f.about}` +
      ` — residual ${fmt(f.residual_ms2, 2)} m/s², 2nd-axis coverage ${fmt(100 * f.axis_coverage, 0)}%`);
    if (f.weak) bits.push("⚠ weak excitation: rocking about a single axis leaves the fit " +
      "unconstrained along it — rock about two clearly different axes, harder");
  }
  if (m.lever_disagreement_m !== null && m.lever_disagreement_m !== undefined)
    bits.push(`CAD vs fit differ by ${(m.lever_disagreement_m * 1000).toFixed(0)} mm` +
      " (expected if the hang point is not the base centre)");
  $("lever-compare").innerHTML = bits.join("<br>");

  const key = JSON.stringify([m.R_chip_to_body, m.lever_active, m.calibrated]);
  if (key !== SNS.mountKey) { SNS.mountKey = key; refreshFrameView(); }
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

/** Draw the base frame at the origin and the IMU's own axes at the lever arm, both in the base
 *  body frame. The mesh's body origin IS the base reference (the model puts its geom at pos 0).
 *  Cheap enough to re-run on the live tick so the measured up-vector animates. */
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

  const r = m.lever_active || m.lever_cad || null;
  if (r) {
    segs.push({ a: [0, 0, 0], b: r, color: [0.75, 0.75, 0.80] });     // base centre -> IMU
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
    (r ? key([0.75, 0.75, 0.8], "base → IMU") + key([1, 0.82, 0.25], "measured up") : "") +
    (r ? "" : "<span class='hint'>enter or fit a lever arm to place the IMU</span>");
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
$("btn-cap-forward").onclick = () =>
  capture("forward", "btn-cap-forward", "Capturing the nose-down tilt — hold it still…");

$("sns-fwd-axis").onchange = (e) =>
  api("/api/sensors/mount", { json: { forward_axis: e.target.value } }).catch(() => {});
$("sns-lever-use").onchange = (e) =>
  api("/api/sensors/mount", { json: { lever_use: e.target.value } }).catch(() => {});

["lev-x", "lev-y", "lev-z"].forEach((id) => { $(id).oninput = () => { SNS.levEdited = true; }; });
$("btn-lever-save").onclick = async () => {
  const v = ["lev-x", "lev-y", "lev-z"].map((id) => parseFloat($(id).value));
  if (v.some((x) => !Number.isFinite(x))) { setBanner("enter all three lever-arm components (metres)", "error", 4000); return; }
  await api("/api/sensors/mount", { json: { lever_cad: v } });
  SNS.levEdited = false;
  setBanner("CAD lever arm saved", "", 2000);
};

$("btn-lever-fit").onclick = async () => {
  if (!SNS.levering) {
    await api("/api/sensors/lever", { json: { action: "start" } });
    setBanner("Recording — rock the robot by hand about TWO different axes, then press stop.", "", 6000);
  } else {
    const d = await api("/api/sensors/lever", { json: { action: "stop" } });
    if (d && d.fit) setBanner(`Lever fit: [${d.fit.r.map((x) => x.toFixed(3)).join(", ")}] m`, "", 5000);
  }
};

$("btn-mount-reset").onclick = async () => {
  await api("/api/sensors/mount/reset", { json: {} });
  SNS.levEdited = false;
  setBanner("Mount calibration reset — values are back in chip axes", "warn", 4000);
};

$("frame-showmesh").onchange = refreshFrameView;
$("frame-showlive").onchange = refreshFrameView;

for (const [id, pose] of [["btn-mock-still", "still"], ["btn-mock-tilt", "tilt"], ["btn-mock-rock", "rock"]])
  $(id).onclick = () => api("/api/mock/sensors", { json: { pose } }).catch(() => {});

buildSensorRows();
pollSensors();
setInterval(pollSensors, 100);
setInterval(drawSensorCharts, 120);
