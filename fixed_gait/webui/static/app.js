/* SpiderBot web UI — vanilla JS, no external assets.
 * Data flow: /api/state @2 Hz (modes, calibration, files), /api/telemetry @10 Hz (samples,
 * linkage). All angles displayed/edited are NORMALIZED degrees (zero pose = 0). */
"use strict";

const MOTORS = ["right.abd", "right.cam", "right.thigh", "left.abd", "left.cam", "left.thigh"];
const ROLES = ["abd", "cam", "thigh"];
const HARD = { abd: 48, cam: 88, thigh: 62 };
const COLORS = { pos: "#4da3ff", cur: "#e0a020", temp: "#e04545", good: "#2c9e3f",
  region: "rgba(44,158,63,0.55)", samples: "rgba(160,170,185,0.4)", gait: "#ff35c8",
  stroke: "#ffd23f", zero: "#ffffff" };

const S = {
  state: null, token: null, seq: 0,
  wsLeg: "right", trajLeg: "right",
  ws: null,                      // /api/workspace payload
  traj: null, trajName: null,    // shown trajectory
  latest: {},                    // motor -> latest telemetry values
  linkage: { left: null, right: null }, linkPrev: { left: null, right: null }, linkT: 0,
  mockTimers: {},
  preview: { on: false, t0: 0 },  // client-side both-legs gait preview animation
  fkmInit: false,                 // sign-map selects synced from state once
};

/* ================================================================ tiny helpers */
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (+v).toFixed(d);

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
  const cal = st.calibration || {};
  const cb = $("calib-badge");
  cb.textContent = cal.stage === "complete" ? "calibrated" : "NOT CALIBRATED";
  cb.className = "badge " + (cal.stage === "complete" ? "cal-ok" : "cal-no");
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
  updateFileLists(st);
  syncFkMapSelects(st);
  $("panel-mock").classList.toggle("hidden", !st.mock);
  $("btn-estop").textContent = (st.estop && st.estop.latched) ? "CLEAR E-STOP" : "E-STOP";
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
  "right.abd": "Push the RIGHT leg outward/inward per the URDF +roll direction.",
  "right.cam": "Backdrive the RIGHT cam in its + direction (same sense as +cam in sim).",
  "right.thigh": "Swing the RIGHT thigh FORWARD (URDF + direction).",
  "left.abd": "Push the LEFT leg per the URDF +roll direction.",
  "left.cam": "Backdrive the LEFT cam in its + direction (same sense as +cam in sim).",
  "left.thigh": "Swing the LEFT thigh FORWARD (URDF + direction).",
};

function updateWizard(st) {
  const cal = st.calibration || {};
  const complete = cal.stage === "complete";
  $("wiz-steps").classList.toggle("hidden", complete);
  $("wiz-summary").classList.toggle("hidden", !complete);
  $("panel-calib").classList.toggle("attention", !complete);
  // gate every actionable panel until calibrated (telemetry + mock stay usable — the wizard
  // itself needs live values and mock dragging)
  document.querySelectorAll("#main > .panel").forEach((p) => {
    if (!["panel-calib", "panel-telemetry", "panel-mock"].includes(p.id))
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
            <button class="btn small" data-flip="${n}">↔ Flip</button>
            <button class="btn small primary" data-confirm="${n}">✓ Confirm</button>
          </div>
        </div>`).join("");
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
class StripChart {
  constructor(canvas, color, span = 60) {
    this.cv = canvas; this.color = color; this.span = span;
    this.t = []; this.v = [];
  }
  push(t, v) {
    if (v === null || v === undefined) return;
    this.t.push(t); this.v.push(v);
    const cut = t - this.span;
    while (this.t.length && this.t[0] < cut) { this.t.shift(); this.v.shift(); }
  }
  draw(now, refLine = null) {
    const c = this.cv, g = c.getContext("2d");
    const w = c.width, h = c.height;
    g.clearRect(0, 0, w, h);
    if (this.v.length < 2) return;
    let lo = Math.min(...this.v), hi = Math.max(...this.v);
    if (refLine !== null) { lo = Math.min(lo, refLine - 5); hi = Math.max(hi, refLine + 5); }
    if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
    if (refLine !== null && refLine <= hi && refLine >= lo) {
      const y = h - ((refLine - lo) / (hi - lo)) * h;
      g.strokeStyle = "rgba(224,69,69,.5)"; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); g.setLineDash([]);
    }
    g.strokeStyle = this.color; g.lineWidth = 1.2; g.beginPath();
    for (let i = 0; i < this.t.length; i++) {
      const x = ((this.t[i] - now + this.span) / this.span) * w;
      const y = h - ((this.v[i] - lo) / (hi - lo)) * h;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.stroke();
    g.fillStyle = "rgba(139,151,168,.9)"; g.font = "9px monospace";
    g.fillText(hi.toFixed(1), 2, 9); g.fillText(lo.toFixed(1), 2, h - 2);
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
      <canvas class="c-pos" width="220" height="34"></canvas>
      <canvas class="c-cur" width="220" height="34"></canvas>
      <canvas class="c-temp" width="220" height="34"></canvas>
    </div>`).join("");
  for (const n of MOTORS) {
    const card = $("mc-" + n.replace(".", "-"));
    charts[n] = {
      pos: new StripChart(card.querySelector(".c-pos"), COLORS.pos),
      cur: new StripChart(card.querySelector(".c-cur"), COLORS.cur),
      temp: new StripChart(card.querySelector(".c-temp"), COLORS.temp),
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
        charts[name].pos.push(d.t[i], m.pos_norm[i]);
        charts[name].cur.push(d.t[i], m.cur[i]);
        charts[name].temp.push(d.t[i], m.temp[i]);
      }
      if (n) S.latest[name] = { pos_norm: m.pos_norm[n - 1], pos_raw: m.pos_raw[n - 1],
        cur: m.cur[n - 1], temp: m.temp[n - 1] };
    }
    if (n) S.lastT = d.t[n - 1];
    for (const side of ["left", "right"]) {
      if (d.linkage[side]) { S.linkPrev[side] = S.linkage[side]; S.linkage[side] = d.linkage[side]; S.linkT = performance.now(); }
    }
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
const manualDesired = {}; let manualDirty = false;
function buildManualRows() {
  $("manual-rows").innerHTML = MOTORS.map((n) => {
    const role = n.split(".")[1];
    return `
    <div class="man-row" id="man-${n.replace(".", "-")}">
      <span class="mr-name">${n}</span>
      <input type="range" class="mr-slider" min="${-HARD[role]}" max="${HARD[role]}" step="0.5" value="0">
      <input type="number" class="num mr-num" step="0.5" value="0">
      <span class="man-sine">
        <label><input type="checkbox" class="sn-en">sine</label>
        <input type="number" class="num sn-a" value="-10" title="angle A °">↔
        <input type="number" class="num sn-b" value="10" title="angle B °">
        <input type="number" class="num sn-f" value="0.3" step="0.05" min="0.02" max="3" title="Hz">Hz
      </span>
    </div>`;
  }).join("");
  for (const n of MOTORS) {
    const row = $("man-" + n.replace(".", "-"));
    const slider = row.querySelector(".mr-slider"), num = row.querySelector(".mr-num");
    const set = (v) => { manualDesired[n] = +v; manualDirty = true; slider.value = v; num.value = v; };
    slider.oninput = () => set(slider.value);
    num.onchange = () => set(num.value);
    const sineSend = () => api("/api/sine", { json: {
      actuator: n, enabled: row.querySelector(".sn-en").checked,
      a_deg: +row.querySelector(".sn-a").value, b_deg: +row.querySelector(".sn-b").value,
      freq_hz: +row.querySelector(".sn-f").value } });
    row.querySelector(".sn-en").onchange = sineSend;
    row.querySelectorAll(".sn-a,.sn-b,.sn-f").forEach((i) => i.onchange = () => {
      if (row.querySelector(".sn-en").checked) sineSend();
    });
  }
}
setInterval(() => {         // 20 Hz slider flush (one final value lands after release too)
  if (!manualDirty) return;
  manualDirty = false;
  api("/api/manual", { json: { targets: { ...manualDesired },
    override: $("chk-override").checked, slew_dps: +$("inp-slew").value } }).catch(() => {});
}, 50);

$("btn-hold").onclick = async () => {
  // enter manual at the current pose: seed sliders from live positions
  for (const n of MOTORS) {
    const v = S.latest[n] ? S.latest[n].pos_norm : 0;
    const row = $("man-" + n.replace(".", "-"));
    row.querySelector(".mr-slider").value = v;
    row.querySelector(".mr-num").value = v;
    manualDesired[n] = v;
  }
  await api("/api/manual", { json: { targets: { ...manualDesired } } });
};
$("btn-release").onclick = () => api("/api/manual/release", { method: "POST" });
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
};

function wsLegData() { return S.ws && S.ws.legs ? S.ws.legs[S.wsLeg] : null; }

function loadWsIntoEditor() {
  const d = wsLegData();
  if (!d) { wsEd.grid = null; wsEd.view.render(); drawAbd(); updateWsStats(); return; }
  const k = d.knee;
  wsEd.shape = k.shape; wsEd.camO = k.cam_origin; wsEd.thighO = k.thigh_origin; wsEd.res = k.res_deg;
  wsEd.grid = unpackBits(k.grid_b64, k.shape[0] * k.shape[1]);
  wsEd.undo = []; wsEd.redo = []; wsEd.dirty = false;
  wsEd.view.fit(k.cam_origin, k.cam_origin + k.shape[0] * k.res_deg,
                k.thigh_origin, k.thigh_origin + k.shape[1] * k.res_deg);
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
  const v = wsEd.view, g = v.g, cv = v.cv;
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
  if (!wsEd.grid) { $("ws-stats").textContent = "no workspace for this leg yet — import or record a sweep"; return; }
  let n = 0; for (let i = 0; i < wsEd.grid.length; i++) n += wsEd.grid[i];
  $("ws-stats").innerHTML = `${n} / ${wsEd.grid.length} cells safe (${wsEd.res}°/cell)` +
    (n === 0 ? ' — <b style="color:#e04545">EMPTY: nothing will pass the safety check!</b>' : "") +
    (wsEd.dirty ? ' — <b style="color:#e0a020">unapplied edits</b>' : "");
}

function wsCellAt(wx, wy) {
  const i = Math.floor((wx - wsEd.camO) / wsEd.res);
  const j = Math.floor((wy - wsEd.thighO) / wsEd.res);
  const [nc, nt] = wsEd.shape;
  return (i >= 0 && i < nc && j >= 0 && j < nt) ? [i, j] : null;
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

function pushUndo() {
  wsEd.undo.push(wsEd.grid.slice());
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
      if (wsEd.tool === "fill") { wsFloodFill(cell); v.render(); updateWsStats(); }
      else { wsApplyBrush(cell, wsEd.tool === "draw" ? 1 : 0); v.render(); }
    },
    onStrokeMove: (cell, prev) => { if (!wsEd.grid || wsEd.tool === "fill") return;
      // interpolate cells between events so fast strokes don't gap
      const steps = Math.max(Math.abs(cell[0] - prev[0]), Math.abs(cell[1] - prev[1]), 1);
      for (let s = 1; s <= steps; s++) {
        const i = Math.round(prev[0] + (cell[0] - prev[0]) * s / steps);
        const j = Math.round(prev[1] + (cell[1] - prev[1]) * s / steps);
        wsApplyBrush([i, j], wsEd.tool === "draw" ? 1 : 0);
      }
      v.render();
    },
    onStrokeEnd: () => updateWsStats(),
    cellAt: wsCellAt,
  });
  $("ws-toolbar").querySelectorAll(".tool").forEach((b) => b.onclick = () => {
    $("ws-toolbar").querySelectorAll(".tool").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    wsEd.tool = b.dataset.tool;
  });
  $("btn-undo").onclick = () => { if (wsEd.undo.length) { wsEd.redo.push(wsEd.grid); wsEd.grid = wsEd.undo.pop(); wsEd.dirty = true; v.render(); updateWsStats(); } };
  $("btn-redo").onclick = () => { if (wsEd.redo.length) { wsEd.undo.push(wsEd.grid); wsEd.grid = wsEd.redo.pop(); wsEd.dirty = true; v.render(); updateWsStats(); } };
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
  S.ws = await (await fetch("/api/workspace")).json();
  $("ws-source").textContent = S.ws.source ? "· " + S.ws.source : "";
  loadWsIntoEditor();
  updateManualRanges();
  renderEE();
}

/* slider bounds follow the demonstrated workspace (the real ranges can exceed the URDF guesses) */
function updateManualRanges() {
  for (const n of MOTORS) {
    const [side, role] = n.split(".");
    const d = S.ws && S.ws.legs ? S.ws.legs[side] : null;
    let lo = -HARD[role], hi = HARD[role];
    if (d) {
      if (role === "abd") { lo = Math.min(lo, d.abd_observed[0] - 10); hi = Math.max(hi, d.abd_observed[1] + 10); }
      else {
        const k = d.knee;
        const o = role === "cam" ? k.cam_origin : k.thigh_origin;
        const nc = role === "cam" ? k.shape[0] : k.shape[1];
        lo = Math.min(lo, o - 10); hi = Math.max(hi, o + nc * k.res_deg + 10);
      }
    }
    const row = $("man-" + n.replace(".", "-"));
    row.querySelector(".mr-slider").min = lo;
    row.querySelector(".mr-slider").max = hi;
  }
}
$("ws-tabs").querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  $("ws-tabs").querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  S.wsLeg = b.dataset.leg;
  $("btn-ws-mirror").textContent =
    S.wsLeg === "right" ? "⇄ copy right → left" : "⇄ copy left → right";
  loadWsIntoEditor();
});
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

$("btn-wsrec-mode").onclick = () => api("/api/record/mode", { json: { kind: "workspace" } });
$("btn-wsrec-take").onclick = () => {
  const active = S.state.recording && S.state.recording.active;
  api("/api/record/take", { json: { leg: S.wsLeg, action: active ? "stop" : "start" } });
};
$("btn-wsrec-undo").onclick = () => api("/api/record/undo", { json: { leg: S.wsLeg } });
$("btn-wsrec-process").onclick = () => api("/api/workspace/process", { json: {
  leg: S.wsLeg, margin_deg: +$("wsrec-margin").value, grid_deg: +$("wsrec-grid").value,
  dilate_deg: +$("wsrec-dilate").value } }).then(refreshWorkspace);

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
  const period = Math.max(0.5, +$("pb-period").value || 8);
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
  const v = trEd.view, g = v.g, cv = v.cv;
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

$("traj-tabs").querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  $("traj-tabs").querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  S.trajLeg = b.dataset.leg;
  trEd.stroke = [];
  fitTrajView();
  trEd.view.render();
});
function fitTrajView() {
  const d = S.ws && S.ws.legs ? S.ws.legs[S.trajLeg] : null;
  if (d) {
    const k = d.knee;
    trEd.view.fit(k.cam_origin, k.cam_origin + k.shape[0] * k.res_deg,
                  k.thigh_origin, k.thigh_origin + k.shape[1] * k.res_deg);
  } else trEd.view.fit(-60, 60, -60, 60);
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
  setBanner("gait processed + saved as " + $("rec-name").value, "", 3500);
  trEd.view.render(); wsEd.view.render(); renderEE();
};

function updateRecordUI(st) {
  const r = st.recording || {};
  const inGait = st.mode === "RECORD_GAIT", inWs = st.mode === "RECORD_WS";
  $("btn-rec-take").disabled = !inGait;
  $("btn-rec-center").disabled = !inGait;
  $("btn-rec-undo").disabled = !inGait || r.active;
  $("btn-rec-take").textContent = (inGait && r.active) ? `■ stop take (${r.n_samples})` : "▶ start take";
  $("btn-rec-take").classList.toggle("active-rec", inGait && r.active);
  $("rec-status").textContent = inGait ?
    `takes R:${r.takes.right} L:${r.takes.left} · center R:${r.centers.right ? "set" : "—"} L:${r.centers.left ? "set" : "—"}` +
    (r.outside_workspace ? " · ⚠ OUTSIDE WORKSPACE" : "") : "";
  $("btn-wsrec-take").disabled = !inWs;
  $("btn-wsrec-undo").disabled = !inWs || r.active;
  $("btn-wsrec-process").disabled = !(r.segments && (r.segments.right || r.segments.left));
  $("btn-wsrec-take").textContent = (inWs && r.active) ? `■ stop segment (${r.n_samples})` : "▶ start segment";
  $("btn-wsrec-take").classList.toggle("active-rec", inWs && r.active);
  $("wsrec-status").textContent = inWs ?
    `segments R:${r.segments.right} L:${r.segments.left}` +
    (r.outside_workspace ? " · ⚠ outside current workspace" : "") : "";
}

/* ---------------- trajectory files ---------------- */
async function showTrajectory(name) {
  try {
    const d = await (await fetch(`/api/trajectory?name=${encodeURIComponent(name)}`)).json();
    if (d.error) { setBanner(d.error, "error", 6000); return; }
    S.traj = d; S.trajName = name;
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
}
function fillSelect(sel, items) {
  const cur = sel.value;
  const want = items.join("|");
  if (sel.dataset.items === want) return;
  sel.dataset.items = want;
  sel.innerHTML = items.map((f) => `<option>${f}</option>`).join("");
  if (items.includes(cur)) sel.value = cur;
}

/* ================================================================ EE animation */
const eeView = { left: null, right: null };   // cached fit per side

function renderEE() {
  for (const side of ["left", "right"]) drawEESide(side);
}

function drawEESide(side) {
  const cv = $("ee-" + side), g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  const legWs = S.ws && S.ws.legs ? S.ws.legs[side] : null;
  const region = legWs && legWs.ee_region;
  const fkOk = S.state && S.state.fk && S.state.fk.available;
  const verified = fkOk && S.state.fk.verified[side];
  if (!verified) {
    g.fillStyle = "#8b97a8"; g.font = "12px sans-serif"; g.textAlign = "center";
    g.fillText(fkOk ? "FK sign map not verified for this side" : "no FK LUT (generate on desktop)",
      cv.width / 2, cv.height / 2 - 8);
    g.fillText(fkOk ? "click 'verify FK sign map', or set signs + 'force enable' below"
                    : "mujoco/spiderbot/gen_fk_lut.py — hot-loads once copied here",
      cv.width / 2, cv.height / 2 + 10);
    g.textAlign = "left";
    return;
  }
  // view fit: region bounds + hip origin
  let xs = [0], ys = [0];
  if (region) for (const p of region) { xs.push(p[0]); ys.push(p[1]); }
  const link = S.linkage[side];
  if (link && link.nodes) for (const p of link.nodes) { xs.push(p[0]); ys.push(p[1]); }
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const mirror = side === "right" ? -1 : 1;
  const pad = 0.06 * Math.max(xmax - xmin, ymax - ymin, 0.2);
  const scale = Math.min(cv.width / (xmax - xmin + 2 * pad), cv.height / (ymax - ymin + 2 * pad));
  const P = (x, z) => {
    x *= mirror;
    const wx0 = mirror === 1 ? xmin : -xmax;
    return [(x - wx0 + pad) * scale, cv.height - (z - ymin + pad) * scale];
  };
  // workspace region
  if (region) {
    g.fillStyle = COLORS.region;
    for (const p of region) { const [x, y] = P(p[0], p[1]); g.fillRect(x - 1.2, y - 1.2, 2.4, 2.4); }
  }
  // gait EE path
  const tr = S.traj && S.traj[side];
  if (tr && tr.ee_path) {
    g.strokeStyle = TRAJ_COLORS[side]; g.lineWidth = 2; g.beginPath();
    let started = false;
    for (const p of tr.ee_path) {
      if (!p) { started = false; continue; }
      const [x, y] = P(p[0], p[1]);
      started ? g.lineTo(x, y) : g.moveTo(x, y);
      started = true;
    }
    g.stroke();
    if (S.preview.on) {                          // preview marker running along the foot path
      const p = tr.ee_path[previewIdx(tr, side, previewPhase())];
      if (p) {
        const [x, y] = P(p[0], p[1]);
        g.fillStyle = TRAJ_COLORS[side]; g.strokeStyle = "#fff"; g.lineWidth = 1.5;
        g.beginPath(); g.arc(x, y, 7, 0, 7); g.fill(); g.stroke();
      }
    }
  }
  // zero EE marker
  if (legWs && legWs.ee_zero) {
    const [x, y] = P(legWs.ee_zero[0], legWs.ee_zero[1]);
    g.strokeStyle = "#fff"; g.lineWidth = 1.5;
    g.beginPath(); g.arc(x, y, 6, 0, 7); g.stroke();
  }
  // hip origin
  g.fillStyle = "#000"; g.strokeStyle = "#fff";
  const [hx, hy] = P(0, 0);
  g.fillRect(hx - 5, hy - 5, 10, 10); g.strokeRect(hx - 5, hy - 5, 10, 10);
  // live linkage — draw_pose style (plot_reachability.py:467-486)
  if (link && link.nodes) {
    const N = {}; ["cam", "thigh", "push", "knee", "ank", "ptip", "ee"]
      .forEach((k, i) => N[k] = link.nodes[i]);
    const line = (pts, color, lw) => {
      g.strokeStyle = color; g.lineWidth = lw; g.beginPath();
      pts.forEach((p, i) => { const [x, y] = P(p[0], p[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke();
    };
    const alpha = link.valid ? 1 : 0.35;
    g.globalAlpha = alpha;
    line([N.cam, N.thigh], "#7a8494", 7);                       // rigid hip block
    line([N.thigh, N.knee, N.ank, N.ee], "#4da3ff", 4);         // serial leg
    line([N.cam, N.push, N.ptip], "#e04545", 3);                // cam/pushrod loop
    for (const k of ["cam", "thigh", "push", "knee", "ank"]) {
      const [x, y] = P(N[k][0], N[k][1]);
      g.fillStyle = "#fff"; g.beginPath(); g.arc(x, y, 3.5, 0, 7); g.fill();
    }
    const [ex, ey] = P(N.ee[0], N.ee[1]);
    g.fillStyle = "#000"; g.strokeStyle = "#fff"; g.lineWidth = 1.5;
    g.beginPath(); g.arc(ex, ey, 6, 0, 7); g.fill(); g.stroke();
    g.globalAlpha = 1;
    if (!link.valid) {
      g.fillStyle = "#e0a020"; g.font = "11px sans-serif";
      g.fillText("⚠ pose outside valid FK region", 8, 14);
    }
  }
  g.fillStyle = "rgba(139,151,168,.8)"; g.font = "10px monospace";
  g.fillText("X fwd →  Z up ↑ (m, rel. hip)", 8, cv.height - 6);
}

$("btn-fk-verify").onclick = async () => {
  const d = await api("/api/fk/verify", { method: "POST" });
  const r = d.report || {};
  $("ee-status").textContent = ["left", "right"].map((s) => {
    const e = r[s] || {};
    if (e.error) return `${s}: ${e.error}`;
    return `${s}: best signs ${e.best} coverage ${(e.coverage * 100).toFixed(0)}%` +
      (e.decisive ? " ✓ verified" : " — NOT decisive → use 'force enable' below if you know the signs");
  }).join("  ·  ");
  await refreshEEData();
};

async function refreshEEData() {
  await pollState();
  await refreshWorkspace();
  if (S.trajName) showTrajectory(S.trajName);
}

for (const side of ["left", "right"]) {
  $(`btn-fkm-${side}`).onclick = async () => {
    const cam = +$(`fkm-${side}-cam`).value, thigh = +$(`fkm-${side}-thigh`).value;
    if (!confirm(`Force-enable the ${side} EE display with signs cam=${cam}, thigh=${thigh}?\n` +
                 "Only do this if you know the mapping — a wrong sign draws a mirrored linkage " +
                 "(display only; the workspace safety check is unaffected).")) return;
    await api("/api/fk/map", { json: { side, cam, thigh, verified: true } });
    setBanner(`${side} EE display enabled (cam=${cam}, thigh=${thigh})`, "", 3500);
    await refreshEEData();
  };
}

function syncFkMapSelects(st) {
  if (S.fkmInit || !st.fk || !st.fk.available || !st.fk.model_map) return;
  S.fkmInit = true;
  for (const side of ["left", "right"]) {
    const m = st.fk.model_map[side] || {};
    $(`fkm-${side}-cam`).value = (m.cam >= 0 ? "+1" : "-1");
    $(`fkm-${side}-thigh`).value = (m.thigh >= 0 ? "+1" : "-1");
  }
}

/* ================================================================ playback */
$("pb-period").oninput = () => $("pb-period-val").textContent = $("pb-period").value;
$("pb-leftphase").oninput = () => $("pb-leftphase-val").textContent = $("pb-leftphase").value;
$("pb-mode").onchange = () =>
  $("pb-current-params").style.display = $("pb-mode").value === "current" ? "" : "none";
$("pb-mode").onchange();

$("btn-pb-start").onclick = () => api("/api/playback/start", { json: {
  name: $("pb-file").value, legs: $("pb-legs").value, mode: $("pb-mode").value,
  period: +$("pb-period").value, left_phase: +$("pb-leftphase").value,
  current_limit: +$("pb-ilimit").value, kp: +$("pb-kp").value, ki: +$("pb-ki").value,
  kd: +$("pb-kd").value, ramp: +$("pb-ramp").value } });
$("btn-pb-stop").onclick = () => api("/api/playback/stop", { method: "POST" });

let pbPatchTimer = null;
function schedulePatch() {
  if (!(S.state && S.state.playback && S.state.playback.running)) return;
  if (pbPatchTimer) clearTimeout(pbPatchTimer);
  pbPatchTimer = setTimeout(() => api("/api/playback", { method: "PATCH", json: {
    period: +$("pb-period").value, left_phase: +$("pb-leftphase").value } }), 250);
}
$("pb-period").addEventListener("input", schedulePatch);
$("pb-leftphase").addEventListener("input", schedulePatch);

function updatePlaybackUI(st) {
  const pb = st.playback;
  $("pb-phase").style.width = pb && pb.running ? (pb.phase * 100) + "%" : "0%";
  $("btn-pb-start").disabled = !!(pb && pb.running);
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
  setupWsCanvas();
  setupTrajCanvas();
  refreshWorkspace().then(fitTrajView);
  pollState();
  setInterval(pollState, 500);
  setInterval(pollTelemetry, 100);
  setInterval(() => { if (!document.hidden) {
    drawCharts();
    wsEd.view.render();
    trEd.view.render();
    drawAbd();
    renderEE();
  } }, 120);
}
boot();
