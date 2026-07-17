/* RL run orchestrator — vanilla JS, no external assets.
 * Polling: /api/state @ 2 s (runs table); for the SELECTED run only:
 * /progress @ 5 s (incremental, since=row index), /log @ 3 s (byte offset).
 * Charts are plain canvas polylines; "sps_inst" is DERIVED client-side from
 * delta(time/total_timesteps)/delta(time/time_elapsed) — time/fps is a cumulative
 * average and is never shown as current speed. */
"use strict";

const DEFAULT_PANELS = ["rollout/ep_rew_mean", "rollout/ep_len_mean", "sps_inst",
                        "reward_terms/air_time"];

const S = {
  state: null,            // /api/state payload
  sel: null,              // selected run name
  detail: null,           // /api/runs/NAME payload
  header: [],  rows: [],  next: 0,      // accumulated progress rows for sel
  panels: DEFAULT_PANELS.slice(),
  logOffset: 0,
  progBusy: false,
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
  return (+v).toFixed(a >= 100 ? 0 : d);
}

let bannerTimer = null;
function banner(msg, ms = 6000) {
  const b = $("banner");
  if (!msg) { b.classList.add("hidden"); return; }
  b.textContent = msg;
  b.classList.remove("hidden");
  if (bannerTimer) clearTimeout(bannerTimer);
  if (ms) bannerTimer = setTimeout(() => b.classList.add("hidden"), ms);
}

async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) {
    o.method = "POST";
    o.headers = { "Content-Type": "application/json" };
    o.body = JSON.stringify(o.json);
    delete o.json;
  }
  const r = await fetch(path, o);
  const d = await r.json();
  if (d && d.ok === false) throw new Error(d.error || "request failed");
  return d;
}

/* ================================================================ state poll (2 s) */
async function pollState() {
  try {
    S.state = await api("/api/state");
    $("conn").classList.add("alive");
    $("conn-label").textContent = `${S.state.runs.length} runs`;
  } catch (e) {
    $("conn").classList.remove("alive");
    $("conn-label").textContent = "disconnected";
    return;
  }
  renderRuns();
  fillBasePicker();
  if (S.sel) renderDetailHead();
}

function runByName(name) {
  return (S.state?.runs || []).find((r) => r.name === name) || null;
}

function renderRuns() {
  const body = $("runs-body");
  const rows = S.state.runs.map((r) => {
    const frac = (r.steps && r.total) ? Math.min(1, r.steps / r.total) : 0;
    const probe = r.probe_verdict
      ? `<span class="badge ${/pass/i.test(r.probe_verdict) ? "pass" : "fail"}">${esc(r.probe_verdict)}</span>` : "";
    const killable = r.managed && r.status === "running";
    return `<tr data-run="${esc(r.name)}" class="${r.name === S.sel ? "sel" : ""}">
      <td><div class="run-name">${esc(r.name)}</div>
          <div class="run-src">${esc(r.source.kind)}:${esc(r.source.base ?? "?")}</div></td>
      <td><span class="pill ${esc(r.status)}">${esc(r.status)}</span></td>
      <td class="bar-wrap"><div class="bar-label">${fmt(r.steps, 0)} / ${fmt(r.total, 0)}</div>
          <div class="bar"><i style="width:${(frac * 100).toFixed(1)}%"></i></div></td>
      <td class="num">${fmt(r.ep_rew, 1)}</td>
      <td class="num">${fmt(r.sps_inst, 0)}</td>
      <td>${probe}</td>
      <td>${killable ? `<button class="mini danger" data-kill="${esc(r.name)}">kill</button>` : ""}</td>
    </tr>`;
  });
  body.innerHTML = rows.join("");
}

$("runs-body").addEventListener("click", (ev) => {
  const kill = ev.target.closest("[data-kill]");
  if (kill) { killRun(kill.dataset.kill); ev.stopPropagation(); return; }
  const tr = ev.target.closest("tr[data-run]");
  if (tr) selectRun(tr.dataset.run);
});

async function killRun(name) {
  if (!confirm(`Kill training run '${name}' (taskkill /T /F the whole tree)?`)) return;
  try {
    await api(`/api/runs/${name}/kill`, { json: {} });
    banner(`killed ${name}`, 3000);
  } catch (e) { banner("kill failed: " + e.message); }
  pollState();
}

/* ================================================================ detail panel */
async function selectRun(name) {
  S.sel = name;
  S.header = []; S.rows = []; S.next = 0;
  S.logOffset = 0;
  S.panels = DEFAULT_PANELS.slice();
  $("log").textContent = "";
  $("detail-empty").classList.add("hidden");
  $("detail").classList.remove("hidden");
  $("d-name").textContent = name;
  renderRuns();
  try {
    S.detail = await api(`/api/runs/${name}`);
  } catch (e) { banner("detail failed: " + e.message); return; }
  renderDetailStatic();
  renderDetailHead();
  pollProgress();
  pollLog();
}

function renderDetailStatic() {
  const d = S.detail;
  const src = d.summary?.source || {};
  $("d-source").textContent =
    `${src.kind}:${src.base ?? "?"}   ·   n_envs ${d.n_envs ?? "?"}   ·   total ${fmt(d.total_steps, 0)}`;
  $("d-desc").textContent = d.description || "(no description)";
  const diff = d.config_diff || {};
  const keys = Object.keys(diff).sort();
  $("diff-count").textContent = keys.length;
  $("diff-body").innerHTML = keys.map((k) =>
    `<tr><td>${esc(k)}</td><td class="def">${esc(JSON.stringify(diff[k][0]))}</td>
     <td class="val">${esc(JSON.stringify(diff[k][1]))}</td></tr>`).join("");
}

function renderDetailHead() {
  const r = runByName(S.sel);
  if (!r) return;
  const st = $("d-status");
  st.textContent = r.status;
  st.className = "pill " + r.status;
  const m = (S.state.managed || {})[S.sel];
  const t = m && m.tool;
  $("d-tool").textContent = t
    ? (t.alive ? `${t.kind} running…`
               : `${t.kind} done (rc ${t.rc})${t.error ? " — " + t.error : ""}`) : "";
  $("d-kill").disabled = !(r.managed && r.status === "running");
  const links = [["d-plots", "training_plots.png"], ["d-runmp4", "run.mp4"],
                 ["d-evalmp4", "eval.mp4"], ["d-probejson", "gait_probe.json"]];
  for (const [id, f] of links) {
    const a = $(id);
    const has = (r.artifacts || []).includes(f);
    a.classList.toggle("hidden", !has);
    if (has) a.href = `/api/runs/${S.sel}/file/${f}`;
  }
}

$("d-kill").onclick = () => killRun(S.sel);
$("d-eval").onclick = () => runTool("evaluate");
$("d-probe").onclick = () => runTool("gait_probe");

async function runTool(kind) {
  try {
    await api(`/api/runs/${S.sel}/tool`, { json: { kind } });
    banner(`${kind} started for ${S.sel}`, 3000);
  } catch (e) { banner(`${kind} failed: ` + e.message); }
  pollState();
}

/* ================================================================ progress + charts (5 s) */
async function pollProgress() {
  if (!S.sel || S.progBusy) return;
  S.progBusy = true;
  const run = S.sel;
  try {
    for (let hop = 0; hop < 20; hop++) {          // drain backlog on first load
      const d = await api(`/api/runs/${run}/progress?since=${S.next}`);
      if (run !== S.sel) return;                  // user switched runs mid-fetch
      if (d.header.length) S.header = d.header;
      if (d.rows.length) { S.rows.push(...d.rows); S.next = d.next; }
      if (!d.more) break;
    }
  } catch (e) { /* transient — next poll retries */ }
  finally { S.progBusy = false; }
  if (run === S.sel) { fillChartAdd(); drawCharts(); }
}

function colIdx(name) { return S.header.indexOf(name); }

function series(key) {
  const n = S.rows.length;
  const iTs = colIdx("time/total_timesteps");
  const xs = new Array(n), ys = new Array(n);
  for (let i = 0; i < n; i++) xs[i] = iTs >= 0 ? S.rows[i][iTs] : i;
  if (key === "sps_inst") {
    const iEl = colIdx("time/time_elapsed");
    let pt = null, pe = null;
    for (let i = 0; i < n; i++) {
      const t = iTs >= 0 ? S.rows[i][iTs] : null;
      const e = iEl >= 0 ? S.rows[i][iEl] : null;
      ys[i] = null;
      if (t !== null && e !== null) {
        if (pt !== null && e > pe) ys[i] = (t - pt) / (e - pe);
        pt = t; pe = e;
      }
    }
  } else {
    const ci = colIdx(key);
    for (let i = 0; i < n; i++) ys[i] = ci >= 0 ? S.rows[i][ci] : null;
  }
  return { xs, ys };
}

function fillChartAdd() {
  const sel = $("chart-add");
  const opts = ["sps_inst", ...S.header].filter((h) => h && !S.panels.includes(h));
  const cur = new Set([...sel.options].map((o) => o.value));
  const want = new Set(["", ...opts]);
  if (cur.size === want.size && opts.every((o) => cur.has(o))) return;
  sel.innerHTML = `<option value="">+ add column…</option>` +
    opts.map((h) => `<option value="${esc(h)}">${esc(h)}</option>`).join("");
}

$("chart-add").onchange = () => {
  const v = $("chart-add").value;
  if (v && !S.panels.includes(v)) { S.panels.push(v); drawCharts(); }
  $("chart-add").value = "";
};

function drawCharts() {
  const wrap = $("charts");
  // rebuild panel divs only when the panel list changed
  const have = [...wrap.children].map((c) => c.dataset.key);
  if (have.length !== S.panels.length || have.some((k, i) => k !== S.panels[i])) {
    wrap.innerHTML = "";
    for (const key of S.panels) {
      const div = document.createElement("div");
      div.className = "chart-panel";
      div.dataset.key = key;
      div.innerHTML = `<canvas></canvas><button class="rm" title="remove">×</button>`;
      div.querySelector(".rm").onclick = () => {
        S.panels = S.panels.filter((k) => k !== key);
        drawCharts();
      };
      wrap.appendChild(div);
    }
  }
  for (const div of wrap.children) {
    const { xs, ys } = series(div.dataset.key);
    drawChart(div.querySelector("canvas"), div.dataset.key, xs, ys);
  }
}

function drawChart(canvas, label, xs, ys) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 340, H = canvas.clientHeight || 150;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const ml = 46, mr = 8, mt = 20, mb = 16;   // margins

  const pts = [];
  for (let i = 0; i < ys.length; i++)        // skip nulls
    if (ys[i] !== null && Number.isFinite(ys[i]) && xs[i] !== null) pts.push([xs[i], ys[i]]);
  ctx.font = "10px Consolas, monospace";
  ctx.fillStyle = "#8a94a0";
  if (pts.length === 0) {
    ctx.fillText(label + "  (no data)", ml, 14);
    return;
  }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [x, y] of pts) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.08;
  y0 -= pad; y1 += pad;
  const X = (x) => ml + ((x - x0) / (x1 - x0)) * (W - ml - mr);
  const Y = (y) => H - mb - ((y - y0) / (y1 - y0)) * (H - mt - mb);

  // axes + labels
  ctx.strokeStyle = "#30363d";
  ctx.beginPath();
  ctx.moveTo(ml, mt); ctx.lineTo(ml, H - mb); ctx.lineTo(W - mr, H - mb);
  ctx.stroke();
  ctx.fillStyle = "#8a94a0";
  ctx.textAlign = "right";
  ctx.fillText(fmt(y1), ml - 4, mt + 8);
  ctx.fillText(fmt(y0), ml - 4, H - mb);
  ctx.textAlign = "left";
  ctx.fillText(fmt(x0, 0), ml, H - 4);
  ctx.textAlign = "right";
  ctx.fillText(fmt(x1, 0), W - mr, H - 4);

  // polyline
  ctx.strokeStyle = "#4da3ff";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
  ctx.stroke();

  // title + last-value legend
  ctx.textAlign = "left";
  ctx.fillStyle = "#d8dee6";
  ctx.fillText(label, ml, 13);
  ctx.fillStyle = "#4da3ff";
  ctx.fillText("last " + fmt(pts[pts.length - 1][1], 3), ml + ctx.measureText(label).width + 12, 13);
}

/* ================================================================ log tail (3 s) */
async function pollLog() {
  if (!S.sel) return;
  const run = S.sel;
  let d;
  try { d = await api(`/api/runs/${run}/log?offset=${S.logOffset}`); }
  catch (e) { return; }
  if (run !== S.sel) return;
  if (d.offset < S.logOffset) $("log").textContent = "";   // log restarted
  S.logOffset = d.offset;
  if (d.data) {
    const el = $("log");
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    el.textContent = (el.textContent + d.data).slice(-200000);
    if (atBottom) el.scrollTop = el.scrollHeight;
  }
}

/* ================================================================ launch form */
function fillBasePicker() {
  const sel = $("l-base");
  if (sel.dataset.filled || !S.state) return;
  const p = (S.state.presets || []).map((n) =>
    `<option value="preset:${esc(n)}">preset: ${esc(n)}</option>`).join("");
  const e = (S.state.experiments || []).map((x) =>
    `<option value="experiment:${esc(x.path)}" data-desc="${esc(x.description)}">experiment: ${esc(x.name)}</option>`).join("");
  sel.innerHTML = `<optgroup label="presets (rl/config.py)">${p}</optgroup>` +
                  `<optgroup label="experiments (yaml)">${e}</optgroup>`;
  sel.dataset.filled = "1";
  sel.onchange = () => {
    const o = sel.selectedOptions[0];
    $("l-base-desc").textContent = o ? (o.dataset.desc || "") : "";
  };
}

$("btn-new").onclick = () => {
  $("l-error").classList.add("hidden");
  $("launch-back").classList.remove("hidden");
  $("l-name").focus();
};
$("l-cancel").onclick = () => $("launch-back").classList.add("hidden");
$("launch-back").addEventListener("click", (ev) => {
  if (ev.target === $("launch-back")) $("launch-back").classList.add("hidden");
});

$("l-desc").addEventListener("input", () => {
  $("l-go").disabled = !$("l-desc").value.trim();   // description is REQUIRED
});

function parseOverrides(text) {
  const out = {};
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const i = line.indexOf("=");
    if (i <= 0) throw new Error(`override line needs field=value: "${line}"`);
    const key = line.slice(0, i).trim();
    const vs = line.slice(i + 1).trim();
    try { out[key] = JSON.parse(vs); } catch (_) { out[key] = vs; }   // bare string ok
  }
  return out;
}

$("l-go").onclick = async () => {
  const err = $("l-error");
  err.classList.add("hidden");
  let overrides;
  try { overrides = parseOverrides($("l-ovr").value); }
  catch (e) { err.textContent = e.message; err.classList.remove("hidden"); return; }
  const spec = {
    base: $("l-base").value,
    name: $("l-name").value.trim(),
    description: $("l-desc").value.trim(),
    subproc: $("l-subproc").checked,
    overrides,
  };
  if ($("l-steps").value) spec.steps = +$("l-steps").value;
  if ($("l-nenvs").value) spec.n_envs = +$("l-nenvs").value;
  $("l-go").disabled = true;
  try {
    await api("/api/launch", { json: spec });
    $("launch-back").classList.add("hidden");
    $("l-name").value = ""; $("l-desc").value = ""; $("l-ovr").value = "";
    banner(`launched ${spec.name}`, 3000);
    await pollState();
    selectRun(spec.name);
  } catch (e) {
    err.textContent = e.message;
    err.classList.remove("hidden");
  } finally {
    $("l-go").disabled = !$("l-desc").value.trim();
  }
};

/* ================================================================ boot */
pollState();
setInterval(pollState, 2000);
setInterval(pollProgress, 5000);
setInterval(pollLog, 3000);
window.addEventListener("resize", () => drawCharts());
