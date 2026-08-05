/* Sense HAT (B) panel. Loaded after app.js, reuses its globals (api, $, fmt, StripChart,
 * setBanner). Polls /api/sensors on its own (~10 Hz) rather than riding /api/state: the IMU is a
 * fast signal and the state poll is 2 Hz.
 *
 * Values are shown in the ICM-20948's OWN chip frame (see AXIS_MAP in sensehat.py) — the mounting
 * hint under the horizon reports which chip axis gravity currently sits on so the frame can be
 * checked at a glance. Roll/pitch are gravity-referenced; yaw is gyro-integrated and drifts. */
"use strict";

const SNS = { seq: 0, charts: null, rpy: [0, 0, 0], down: false };

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
  $("sns-status").textContent =
    `${chips}${d.mag_live ? "" : " (mag idle)"}${d.i2c_errors ? ` — ${d.i2c_errors} read errors` : ""}`;
  const bs = d.bias_status || {};
  $("sns-bias").textContent = "gyro zero: " + (bs.msg || "—");
  $("sns-bias").className = "hint" + (bs.state === "moving" ? " warn-text" : "");
  $("sns-mount").textContent = mountHint(v);
}

/** Which chip axis is gravity on right now? The only cheap check that the HAT's mounting matches
 *  what the numbers are labelled with. Only meaningful while the robot is still. */
function mountHint(v) {
  const a = [v.ax, v.ay, v.az];
  if (a.some((x) => x === null || x === undefined)) return "";
  if (Math.abs(v.acc_mag - 1) > 0.08) return "in motion — mounting check needs the robot at rest";
  const i = a.map(Math.abs).indexOf(Math.max(...a.map(Math.abs)));
  const axis = "XYZ"[i], sign = a[i] < 0 ? "−" : "+";
  const level = sign === "+" && axis === "Z";
  return `gravity on ${sign}${axis}` + (level ? " (chip Z up)"
    : " — chip frame is not Z-up; set AXIS_MAP in sensehat.py to map it to the robot's body axes");
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
}

/* ================================================================ actions + boot */
$("btn-gyro-bias").onclick = async () => {
  const b = $("btn-gyro-bias");
  b.disabled = true;
  try {
    await api("/api/sensors/gyro_bias", { json: {} });
    setBanner("Averaging the gyro zero — hold the robot still…", "", 2500);
  } finally {
    setTimeout(() => { b.disabled = false; }, 2000);
  }
};

buildSensorRows();
pollSensors();
setInterval(pollSensors, 100);
setInterval(drawSensorCharts, 120);
