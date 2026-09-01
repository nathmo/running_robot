/* Policy inference panel. Loaded after app.js; reuses its globals (api, $, setBanner).
 *
 * What this panel is and is not: a policy bundle (.npz from robot/deploy/export_policy.py) is
 * the whole control law — estimator + policy MLPs, impedance gains, gait reconstruction,
 * observation statistics — and the panel shows exactly what the selected bundle carries, checks
 * what stands between this robot and a real run, and can prove the export + runtime end to end
 * with a --mock dress rehearsal (mock bus, mock IMU, energises nothing).
 *
 * It DOES run a policy on the motors, and the run happens inside the daemon (mode POLICY) rather
 * than in a subprocess: the CAN bus has exactly one owner, so run_policy.py — which does the same
 * job headless — refuses to start while this daemon is up. Either way the control law and the
 * safety governor are the deploy package's own, imported rather than reimplemented; what this file
 * contributes is an arming form, a dead-man, and a readout.
 */
"use strict";

const POL = { list: [], file: null, pollTimer: null, running: false,
              preflight: [], acks: {}, deadman: null, saved: true, runFile: null, armedAt: 0 };

function polEsc(s) {
  return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
}

function polInit() {
  $("btn-pol-refresh").onclick = polRefresh;
  $("pol-file").onchange = () => { POL.file = $("pol-file").value || null; polInfo(); };
  $("btn-pol-rehearse").onclick = polRehearse;
  $("btn-pol-rehearse-stop").onclick = () => api("/api/policy/rehearse/stop", { json: {} });
  $("pol-upload").onchange = polUpload;
  $("btn-pol-arm").onclick = polArm;
  $("btn-pol-run-stop").onclick = () => polStopRun(false);
  $("btn-pol-run-kill").onclick = () => polStopRun(true);
  $("pol-supported").onchange = polRunButtons;
  // Leaving the page is the dead-man's own signal, so it must not ALSO be the thing that keeps a
  // stale interval alive: the timer checks visibility every tick and simply stops posting.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && polRunPhase()) polKeepaliveStart();
  });
  polRefresh();
  polPollStatus();               // pick up a rehearsal that outlived a page reload
}

function polSelectedValid() {
  return POL.list.some((b) => b.file === POL.file && b.valid);
}

function polButtons() {
  $("btn-pol-rehearse").disabled = POL.running || !polSelectedValid();
  $("btn-pol-rehearse-stop").disabled = !POL.running;
  polRunButtons();
}

async function polRefresh() {
  let d;
  try { d = await api("/api/policy/list"); } catch (_) { return; }
  POL.list = d.bundles || [];
  const sel = $("pol-file");
  const prev = POL.file;
  sel.innerHTML = POL.list.map((b) =>
    `<option value="${polEsc(b.file)}"${b.valid ? "" : " disabled"}>` +
    polEsc(b.valid ? `${b.file} — ${b.run || "?"} / ${b.checkpoint || "?"} [${b.where || "?"}]`
                   : `${b.file} — NOT A BUNDLE [${b.where || "?"}]`) + `</option>`).join("")
    || `<option value="">— no bundles in data/policies/ yet —</option>`;
  const nValid = POL.list.filter((b) => b.valid).length;
  $("pol-count").textContent = `${nValid} bundle${nValid === 1 ? "" : "s"}`;
  if (prev && POL.list.some((b) => b.file === prev)) sel.value = prev;
  POL.file = sel.value || null;
  polInfo();
}

async function polInfo() {
  const el = $("pol-info");
  const pf = $("pol-preflight");
  polButtons();
  const row = POL.list.find((b) => b.file === POL.file);
  if (!row) {
    el.classList.add("hidden");
    pf.innerHTML = "";
    POL.preflight = [];
    polRenderAcks([]);
    $("pol-cmd").textContent = "select a bundle above";
    polRunButtons();
    return;
  }
  if (!row.valid) {
    el.classList.remove("hidden");
    el.innerHTML = `<div class="pol-warn">✗ ${polEsc(row.file)} did not load as a policy ` +
      `bundle: ${polEsc(row.error)}</div>`;
    pf.innerHTML = "";
    POL.preflight = [];
    polRenderAcks([]);
    $("pol-cmd").textContent = "select a bundle above";
    polRunButtons();
    return;
  }
  let d;
  try { d = await api("/api/policy/info", { json: { file: POL.file } }); } catch (_) { return; }
  const i = d.info || {};
  const cmd = i.cmd_box || {};
  el.classList.remove("hidden");
  el.innerHTML =
    `<div class="pol-arch"><b>${polEsc(i.run)} / ${polEsc(i.checkpoint)}</b> — ` +
    `${i.control_hz || "?"} Hz control, ${i.action_dim} action dims` +
    `<div class="ro-grid">` +
    `<div>observation</div><div>${i.frame_dim} × ${i.history_len} frames = ${i.obs_dim}</div>` +
    `<div>estimator MLP</div><div>${(i.estimator || []).join(" → ")}</div>` +
    `<div>policy MLP</div><div>${(i.policy || []).join(" → ")} (obs ‖ estimate)</div>` +
    `<div>impedance kp</div><div>${(i.imp_kp || []).join("–")} N·m/rad base</div>` +
    `<div>impedance kd</div><div>${(i.imp_kd || []).join("–")} N·m·s/rad base</div>` +
    `<div>trained commands</div><div>fwd ${cmd.fwd_ms} m/s · back ${cmd.back_ms} · ` +
    `yaw ${cmd.yaw_rads} rad/s</div>` +
    `</div></div>` +
    (d.warnings || []).map((w) => `<div class="pol-warn">⚠ ${polEsc(w)}</div>`).join("");
  const checks = d.preflight || [];
  POL.preflight = checks;
  polRenderAcks(checks);
  // the command box the policy was actually trained to answer, next to the inputs that set it
  $("pol-cmd-box").textContent = `trained box: ${cmd.back_ms == null ? "?" : -cmd.back_ms}` +
    ` … ${cmd.fwd_ms} m/s, ±${cmd.yaw_rads} rad/s`;
  const bad = checks.filter((c) => !c.ok);
  pf.innerHTML = `<div class="pol-pf">` + checks.map((c) =>
    c.ok ? `<span class="pf-ok">✓ ${polEsc(c.name)}</span>`
         : `<span class="pf-bad">✗ ${polEsc(c.name)} — ${polEsc(c.why)}</span>`).join("") +
    `</div>` +
    (bad.length ? "" : `<div class="hint">every preflight gate for a real run is green</div>`);
  $("pol-cmd").textContent = d.command || "";
  polRunButtons();
}

async function polRehearse() {
  if (!polSelectedValid()) return;
  try {
    await api("/api/policy/rehearse",
              { json: { file: POL.file, seconds: +$("pol-seconds").value || 5 } });
  } catch (_) { return; }
  POL.running = true;
  polButtons();
  $("pol-log").classList.remove("hidden");
  $("pol-log").textContent = "(starting…)";
  polPollStatus();
}

/* One poll per second while a rehearsal runs; the timer dismantles itself when nothing is. */
function polPollStatus() {
  if (POL.pollTimer) return;
  const tick = async () => {
    let d;
    try { d = await api("/api/policy/rehearse/status"); } catch (_) { return; }
    const r = d.rehearsal;
    const st = $("pol-rehearse-status");
    if (!r) {
      clearInterval(POL.pollTimer);
      POL.pollTimer = null;
      POL.running = false;
      polButtons();
      return;
    }
    $("pol-log").classList.remove("hidden");
    $("pol-log").textContent = r.tail || "(no output yet)";
    $("pol-log").scrollTop = $("pol-log").scrollHeight;
    POL.running = !!r.running;
    polButtons();
    if (r.running) {
      st.textContent = `rehearsing ${r.file} — ${r.elapsed_s.toFixed(0)} s`;
    } else {
      clearInterval(POL.pollTimer);
      POL.pollTimer = null;
      st.textContent = r.returncode === 0
        ? `✓ ${r.file} rehearsed cleanly — the bundle and the runtime agree`
        : `✗ rehearsal exited with code ${r.returncode} — read the log above`;
    }
  };
  POL.pollTimer = setInterval(tick, 1000);
  tick();
}

async function polUpload(ev) {
  const f = ev.target.files[0];
  ev.target.value = "";
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  let r;
  try {
    r = await fetch("/api/policy/upload", { method: "POST", body: fd });
  } catch (e) { setBanner("upload failed: " + e, "error", 6000); return; }
  const d = await r.json();
  if (!d.ok) { setBanner(d.error, "error", 8000); return; }
  setBanner(`bundle ${d.file} uploaded and validated`, "", 3000);
  POL.file = d.file;
  polRefresh();
}

/* ================================================================ running one, for real
 *
 * The run happens in the daemon (mode POLICY), because the CAN bus has exactly one owner and
 * run_policy.py refuses to start while the web UI is up. This half of the panel is therefore an
 * arming form, a dead-man, and a readout — it never computes a command.
 *
 * THE DEAD-MAN is the part worth reading. polKeepalive() posts to /api/policy/keepalive five times
 * a second while a run is active AND the page is VISIBLE. Stop posting — close the tab, switch
 * away, lose the Wi-Fi — and the governor soft-stops within POLICY_DEADMAN_S: the target freezes
 * and the gains bleed out over ~0.3 s, which puts the robot down under control rather than
 * dropping it. It is deliberately not the status poll: a poll proves a browser is alive, which is
 * not the claim being made. The keepalive response carries the full daemon snapshot, so the same
 * request that says "someone is watching" is also what refreshes this readout at 5 Hz.
 *
 * THE ACKNOWLEDGEMENTS below each switch off a check that exists for a reason, so each one is
 * rendered only when its check is actually failing, states what stops being true, and asks before
 * it is armed. `supported` is the exception: it is always required, because it is the one hazard
 * no check in the daemon can see. */
const POL_ACKS = {
  "joint map": {
    key: "skip_jointmap_check",
    label: "run with an UNVERIFIED joint map",
    why: "A sign error in the model→motor map drives every balance correction the WRONG WAY at "
       + "up to 500 N·m/rad, and nothing upstream can see it: the targets stay in range, the "
       + "drives track them, the telemetry looks healthy. The robot simply falls in a way that "
       + "looks like a bad policy. Verify the map (fklut, then make_deploy_map.py) instead.",
  },
  "thermal model": {
    key: "allow_uncalibrated_thermal",
    label: "run with PLACEHOLDER thermal parameters",
    why: "The winding-temperature observer derates every torque budget in the governor. With "
       + "unfitted parameters that estimate is a guess, so the continuous-torque limit is a guess "
       + "— the peak rating and the drive's own phase limit still apply, the thermal one does not.",
  },
  "IMU mount": {
    key: "no_imu",
    label: "run WITHOUT the IMU",
    why: "Gravity is the only fall detector there is. Without it the governor cannot tell an "
       + "upright robot from one on its side, and the tilt kill can never fire. Bench use only, "
       + "with the robot physically restrained.",
  },
};
POL_ACKS["loop rate"] = {
  key: "allow_slow_loop",
  label: "run the control law SLOWER than it was trained to",
  why: "The bundle's control_dt is a constant, so a loop that cannot keep up does not run a "
     + "slightly-degraded policy — it runs a different one: the gait clock advances in slow "
     + "motion and the joint velocities in the observation are inflated by the same ratio, which "
     + "is a state the policy has never seen. It is still bounded by the governor and the "
     + "workspace, so it is a legitimate way to bring the drives up and watch them move. It is "
     + "not a way to evaluate the policy.",
};
POL_ACKS["IMU live"] = POL_ACKS["IMU mount"];
// A "ui:" key is acknowledged HERE and sent NOWHERE: the daemon already has the authoritative
// version of this check and will refuse on its own if it is really unsafe. Ticking it only says
// the operator has read the warning.
POL_ACKS["zeroing freshness"] = {
  key: "ui:stale_zero",
  label: "the drives have NOT been power-cycled since the zero was captured",
  why: "The calibration was loaded from disk rather than captured in this session, and every drive "
     + "re-randomises its raw encoder origin on a power cycle. If they have been power-cycled, "
     + "every joint angle the policy reads is wrong by an unknown offset — press Set Zero. The "
     + "daemon's pre-move guard checks this too and will refuse the run if the raw poses have "
     + "moved; this box only says you have read the warning.",
};

function polRunPhase() {
  const p = (S.state && S.state.policy) || null;
  return p && p.running ? p.phase : null;
}

/* Which preflight failures the operator has explicitly taken on, keyed by ack field. */
/* The acknowledgements that go into the arm request. "ui:" ones are read-and-understood only. */
function polAckState() {
  const on = {};
  document.querySelectorAll("#pol-acks input[type=checkbox]").forEach((el) => {
    if (el.checked && !el.dataset.ack.startsWith("ui:")) on[el.dataset.ack] = true;
  });
  return on;
}

function polRenderAcks(checks) {
  const box = $("pol-acks");
  const bad = (checks || []).filter((c) => !c.ok && POL_ACKS[c.name]);
  const keys = [];
  const html = bad.map((c) => {
    const a = POL_ACKS[c.name];
    if (keys.includes(a.key)) return "";          // IMU mount + IMU live share one acknowledgement
    keys.push(a.key);
    const prev = POL.acks[a.key] ? " checked" : "";
    return `<div class="pol-ack"><label><input type="checkbox" data-ack="${a.key}"${prev}>` +
      `<span><b>${polEsc(a.label)}</b> — ${polEsc(c.why || "")}<br>${polEsc(a.why)}</span>` +
      `</label></div>`;
  }).join("");
  box.innerHTML = html;
  box.querySelectorAll("input[type=checkbox]").forEach((el) => {
    el.onchange = () => {
      if (el.checked && !confirm(
          "You are about to run a policy with a safety check disabled.\n\n" +
          el.parentElement.textContent.trim() +
          "\n\nThis is recorded in the flight recorder. Continue?")) {
        el.checked = false;
      }
      POL.acks[el.dataset.ack] = el.checked;
      polButtons();
    };
  });
}

/* Which preflight failures are NOT overridable — nothing here can be acknowledged away. */
function polBlockers() {
  return (POL.preflight || []).filter((c) => !c.ok && !POL_ACKS[c.name]);
}

function polRunButtons() {
  const phase = polRunPhase();
  const ticked = {};
  document.querySelectorAll("#pol-acks input[type=checkbox]").forEach(
    (el) => { if (el.checked) ticked[el.dataset.ack] = true; });
  const unacked = (POL.preflight || []).filter(
    (c) => !c.ok && POL_ACKS[c.name] && !ticked[POL_ACKS[c.name].key]);
  const ready = polSelectedValid() && $("pol-supported").checked
    && !polBlockers().length && !unacked.length && !phase;
  $("btn-pol-arm").disabled = !ready;
  $("btn-pol-run-stop").disabled = !phase;
  $("btn-pol-run-kill").disabled = !phase;
  const st = $("pol-run-status");
  if (phase) st.textContent = "";
  else if (polBlockers().length)
    st.textContent = "blocked: " + polBlockers().map((c) => c.name).join(", ");
  else if (unacked.length) st.textContent = "acknowledge the checks above to arm";
  else if (!$("pol-supported").checked) st.textContent = "confirm the torso is supported";
  else st.textContent = "";
}

async function polArm() {
  const row = POL.list.find((b) => b.file === POL.file);
  if (!row || !row.valid) return;
  const spec = {
    file: POL.file, supported: true,
    v_cmd: +$("pol-v").value || 0, yaw_cmd: +$("pol-yaw").value || 0,
    max_seconds: +$("pol-secs").value || 10,
    ...polAckState(),
  };
  let d;
  try { d = await api("/api/policy/arm", { json: spec }); } catch (_) { return; }
  const a = d.armed || {};
  if (a.v_cmd_clamped || a.yaw_cmd_clamped) {
    setBanner(`command clamped to the box this checkpoint was trained to: ` +
      `${a.v_cmd.toFixed(2)} m/s, ${a.yaw_cmd.toFixed(2)} rad/s`, "warn", 6000);
    $("pol-v").value = a.v_cmd;
    $("pol-yaw").value = a.yaw_cmd;
  }
  POL.saved = false;
  POL.armedAt = Date.now();
  polKeepaliveStart();
}

/* 5 Hz while the run is live AND the page is visible. The response carries the daemon snapshot,
 * so applyState() (inside api()) drives the readout from the same request. */
function polKeepaliveStart() {
  if (POL.deadman) return;
  POL.deadman = setInterval(async () => {
    // Do not tear down before the run has had a chance to APPEAR. The snapshot the arm response
    // carries is republished at 20 Hz, so for the first moments after arming "no policy in the
    // state" means "not yet", not "over" — and tearing down here would stop refreshing the
    // dead-man on a run that is about to start, which soft-stops it 1.5 s later for no reason.
    if (!polRunPhase() && Date.now() - (POL.armedAt || 0) > 3000) {
      clearInterval(POL.deadman);
      POL.deadman = null;
      polSaveRun();                       // the run ended: keep its 200 Hz log
      return;
    }
    if (document.visibilityState !== "visible") return;   // THIS is the dead-man
    try { await api("/api/policy/keepalive", { json: {} }); } catch (_) { /* poll shows it */ }
  }, 200);
}

async function polStopRun(hard) {
  if (hard && !confirm("Kill the run NOW?\n\nGains go to zero this tick and the robot goes limp " +
                       "wherever it is — it will drop. A soft stop freezes the target and bleeds " +
                       "the gains out over 0.3 s instead, which puts it down under control."))
    return;
  try { await api("/api/policy/stop", { json: { hard: !!hard } }); } catch (_) { /* shown */ }
}

/* Save the finished run's 200 Hz log exactly once. Nothing is lost if this fails — the flight
 * recorder has the same window at the same rate — but this is the file with the policy's own
 * targets, gains and winding estimates in it. */
async function polSaveRun() {
  if (POL.saved) return;
  POL.saved = true;
  try {
    const d = await api("/api/policy/run/save", { json: {} });
    setBanner(`policy run saved: ${d.file} (${d.rows} ticks)`, "", 5000);
    POL.runFile = d.file;
    polRenderRun(S.state);
  } catch (_) { /* 404 = nothing finished; not worth a banner */ }
}

function polFmtTemps(p) {
  const names = p.winding_names || [];
  return (p.winding_c || []).map((t, i) =>
    `<span class="${t >= 90 ? "hot" : ""}">${polEsc(names[i] || i)} <b>${t.toFixed(0)}°</b>` +
    `<span class="hint">/${(p.peak_winding_c || [])[i]}</span></span>`).join("");
}

function polRenderRun(st) {
  const p = st && st.policy;
  const box = $("pol-run");
  if (!p) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.className = "pol-run " + (p.running ? "live" : (p.stop === "running" ? "" : "stopped"));
  const frac = p.phase === "approach" ? p.approach_frac
             : (p.phase === "run" ? Math.min(1, p.elapsed_s / Math.max(p.max_seconds, 1e-6)) : 1);
  const clamps = Object.entries(p.clamps || {}).map(([k, v]) => `${k}×${v}`).join(" · ");
  box.innerHTML =
    `<div class="row"><span class="pol-phase ${polEsc(p.phase)}">${polEsc(p.phase)}</span>` +
    `<b>${polEsc(p.run)}</b><span class="hint">${polEsc(p.file)}</span>` +
    `<span class="hint">cmd ${p.v_cmd.toFixed(2)} m/s · ${p.yaw_cmd.toFixed(2)} rad/s</span></div>` +
    `<div class="pol-bar"><div style="width:${(frac * 100).toFixed(1)}%"></div></div>` +
    `<div class="ro-grid">` +
    `<div>elapsed</div><div>${p.elapsed_s.toFixed(2)} / ${p.max_seconds.toFixed(0)} s` +
      ` · ${p.ticks} ticks, ${p.late_ticks} late</div>` +
    `<div>gait</div><div>${p.gait_freq_hz.toFixed(2)} Hz, phase ${p.gait_phase.toFixed(2)}</div>` +
    `<div>governor</div><div>${polEsc(p.stop)}${clamps ? " · clamped " + polEsc(clamps) : ""}` +
      `${p.ramp < 1 ? ` · gains at ${(p.ramp * 100).toFixed(0)}%` : ""}</div>` +
    `<div>workspace</div><div>${p.ws_blocked_ticks ? p.ws_blocked_ticks +
      " ticks frozen at the safe-workspace edge" : "clear"}` +
      `${p.target_clip_ticks ? " · " + p.target_clip_ticks + " ticks at the ctrl range" : ""}</div>` +
    `<div>feedback</div><div>telemetry ${p.telemetry_age_ms.toFixed(0)} ms · IMU ` +
      `${p.imu_age_ms.toFixed(0)} ms · dead-man ${p.deadman_age_s.toFixed(1)} s</div>` +
    `</div>` +
    `<div class="pol-temps">${polFmtTemps(p)}</div>` +
    ((p.reasons || []).length
      ? `<div class="pol-reasons">■ ${(p.reasons || []).map(polEsc).join("<br>■ ")}</div>` : "") +
    (p.exit_reason && !p.running
      ? `<div class="hint">ended: ${polEsc(p.exit_reason)}` +
        `${p.reached_run ? "" : " (never reached the policy — it stopped during the approach)"}` +
        `${POL.runFile ? ` · <a href="/api/policy/run/download?file=${encodeURIComponent(POL.runFile)}">` +
          `${polEsc(POL.runFile)}</a>` : ""}</div>` : "");
}

/* Called from applyState in app.js on every state update, however it arrived. */
window.onPolicyState = function (st) {
  polRenderRun(st);
  polRunButtons();
  if (polRunPhase() && !POL.deadman) polKeepaliveStart();   // adopt a run started in another tab
};

document.addEventListener("DOMContentLoaded", polInit);
