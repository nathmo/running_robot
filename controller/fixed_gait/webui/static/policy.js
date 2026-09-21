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
              preflight: [], acks: {}, deadman: null, saved: true, runFile: null, armedAt: 0,
              // the selected bundle's command channel: "velocity" (v1), "run_stop" (v2), or
              // "speed" (the joystick lineage — a slider in m/s, see polRenderCommand)
              cmdKind: "velocity", info: null,
              // slider state. `speedPending` + `speedTimer` throttle the POST while dragging;
              // `speedDrag` stops a status poll yanking the knob out from under a finger.
              speedPending: null, speedTimer: null, speedDrag: false, speedRun: null };

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
  $("btn-pol-go").onclick = () => polCommand(true);
  $("btn-pol-halt").onclick = () => polCommand(false);
  const sl = $("pol-speed");
  sl.oninput = polSpeedInput;                        // live while dragging, throttled
  sl.onchange = polSpeedFlush;                       // and always the final value on release
  sl.onpointerdown = () => { POL.speedDrag = true; };
  sl.onpointerup = () => { POL.speedDrag = false; };
  sl.onblur = () => { POL.speedDrag = false; };
  $("btn-pol-zero").onclick = () => polSpeedSet(0);
  $("btn-pol-aim").onclick = polZeroHeading;
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
    polEsc(b.valid ? `${b.file} — ${b.run || "?"} / ${b.checkpoint || "?"} ` +
                     `(v${b.version} · ${b.hz || "?"} Hz) [${b.where || "?"}]`
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
  if (!row || !row.valid) {
    // nothing selected, or it does not load: fall back to the v1 form rather than leaving the
    // previous bundle's command channel on screen
    POL.cmdKind = "velocity";
    POL.info = null;
    $("pol-cmd-velocity").classList.remove("hidden");
    $("pol-command").classList.add("hidden");
  }
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
  POL.info = i;
  POL.cmdKind = i.command_kind || "velocity";
  const obsLine = i.once_dim
    ? `${i.frame_dim} × ${i.history_len} frames + ${i.once_dim} once = ${i.obs_dim}`
    : `${i.frame_dim} × ${i.history_len} frames = ${i.obs_dim}`;
  el.classList.remove("hidden");
  el.innerHTML =
    `<div class="pol-arch"><b>${polEsc(i.run)} / ${polEsc(i.checkpoint)}</b> — ` +
    `v${i.bundle_version} bundle, ${i.control_hz || "?"} Hz control, ${i.action_dim} action dims` +
    `<div class="ro-grid">` +
    `<div>observation</div><div>${obsLine}</div>` +
    `<div>estimator MLP</div><div>${(i.estimator || []).join(" → ")}</div>` +
    `<div>policy MLP</div><div>${(i.policy || []).join(" → ")} (obs ‖ estimate)</div>` +
    `<div>impedance kp</div><div>${(i.imp_kp || []).join("–")} N·m/rad base</div>` +
    `<div>impedance kd</div><div>${(i.imp_kd || []).join("–")} N·m·s/rad base</div>` +
    (POL.cmdKind === "speed"
      ? `<div>command</div><div>speed slider, ${(i.v_min || 0).toFixed(2)} to ` +
          `${(i.v_max || 0).toFixed(2)} m/s — task[0] = v_cmd / ${(i.v_max || 0).toFixed(2)}, ` +
          `and 0 is walk in place</div>` +
        `<div>commanded in training</div><div>${(i.v_trained || []).length
           ? `${i.v_trained[0].toFixed(2)}–${i.v_trained[1].toFixed(2)} m/s at this checkpoint`
           : "unknown"}</div>` +
        `<div>gait spec</div><div>${i.latched_dims} dims latched at each clock wrap, ` +
          `${i.action_dim - i.latched_dims} residual every tick</div>` +
        `<div>odometry</div><div>none — task[1] is reserved and held at 0, so nothing the policy ` +
          `reads is a position</div>`
    : POL.cmdKind === "run_stop"
      ? `<div>command</div><div>run / stop flag${i.has_run_flag ? "" :
           " — <b>constant</b> for this checkpoint (objective ‘speed’)"}</div>` +
        `<div>gait spec</div><div>${i.latched_dims} dims latched at each clock wrap, ` +
          `${i.action_dim - i.latched_dims} residual every tick</div>` +
        `<div>brake schedule</div><div>${i.brake
           ? `${i.brake.window_s} s window, fitted at ${(i.brake.cruise_speed || 0).toFixed(2)} m/s ` +
             `(${polEsc(i.brake.source)})`
           : "<b>none</b> — STOP will be unavailable; fit one with tools/brake_search.py"}</div>` +
        `<div>red lights in training</div><div>${i.stop_trained
           ? `yes (${i.stop_decel_s} s ramp) — but the flag is not the brake; see the warning`
           : "no"}</div>`
      : `<div>trained commands</div><div>fwd ${cmd.fwd_ms} m/s · back ${cmd.back_ms} · ` +
        `yaw ${cmd.yaw_rads} rad/s</div>`) +
    `</div></div>` +
    (d.warnings || []).map((w) => `<div class="pol-warn">⚠ ${polEsc(w)}</div>`).join("");
  const checks = d.preflight || [];
  POL.preflight = checks;
  polRenderAcks(checks);
  // the command the policy was actually trained to answer, next to the inputs that set it
  const vel = POL.cmdKind === "velocity";
  $("pol-cmd-velocity").classList.toggle("hidden", !vel);
  $("pol-cmd-box").textContent = vel
    ? `trained box: ${cmd.back_ms == null ? "?" : -cmd.back_ms} … ${cmd.fwd_ms} m/s, ` +
      `±${cmd.yaw_rads} rad/s`
    : POL.cmdKind === "speed"
    ? `command: a speed slider, ${(i.v_min || 0).toFixed(2)}–${(i.v_max || 0).toFixed(2)} m/s, ` +
      `live once the run is up — it starts at 0, walking in place`
    : "command: RUN / STOP, live, once the run is up — it starts stopped";
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
  // a v2 bundle has no velocity channel, so the request does not carry one: an input that the
  // daemon would ignore is an input that lies about what was asked for
  const spec = {
    file: POL.file, supported: true,
    max_seconds: +$("pol-secs").value || 10,
    ...(POL.cmdKind === "velocity"
        ? { v_cmd: +$("pol-v").value || 0, yaw_cmd: +$("pol-yaw").value || 0 }
        : {}),
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

/* ================================================================ the v2 gait command
 *
 * RUN and STOP are commands to the POLICY, not to the governor: the gains are untouched and the
 * run continues either way. They are NOT the pair above, which takes the machine away from it.
 *
 * STOP IS THE FITTED BRAKE, NOT THE TASK FLAG, and that distinction is the whole story of this
 * lineage. Ten training configurations failed to teach it to brake; the cluster then measured why
 * (walk_v2/README.md, 2026-09-11 17:00):
 *
 *     upright 512/512   with the policy NOT told it has finished
 *     upright   3/512   same control, policy told it has finished
 *
 * The command channel IS the disturbance. So STOP holds the task flag at 1 and overrides the
 * latched gait spec with a 12-number open-loop schedule fitted offline for this exact checkpoint,
 * while the policy keeps contributing its per-tick residual. A bundle without a schedule has no
 * STOP: the daemon refuses rather than doing the thing that drops the robot.
 *
 * A run always comes up stopped, and that is enforced in the daemon rather than here — a browser
 * that fails to send anything must leave the robot holding its stance, not travelling. */
async function polCommand(run) {
  const p = (S.state && S.state.policy) || null;
  if (!run && p && p.braking) return;             // already braking: pressing again is not a restart
  try { await api("/api/policy/command", { json: { run: !!run } }); } catch (_) { /* shown */ }
  polRenderCommand(S.state);
}

/* ================================================================ the joystick
 *
 * The retrained lineage has no green light: task[0] IS the commanded speed over v_max, a number
 * the policy has been trained at every value of, so the panel is a slider in m/s and 0 — walking
 * in place — is both where a run starts and how you stop it. That is the point of the retrain:
 * the flag was a step change onto an input the policy only ever met at the finish line, and the
 * pair of numbers above is what that did.
 *
 * Two rules here, and both are about not turning a slider into a step:
 *   - the POST is throttled while dragging (the control law slews the command anyway, so what the
 *     robot sees is a ramp either way — this just keeps the Pi's HTTP thread out of the CAN loop);
 *   - a status poll never moves the knob while a finger is on it, or while a send is in flight.
 * The run still comes up at 0 whatever the browser does: that is enforced in the daemon. */
function polSpeedInput() {
  const v = +$("pol-speed").value;
  $("pol-speed-val").textContent = `${v.toFixed(2)} m/s`;
  POL.speedPending = v;
  if (POL.speedTimer) return;
  POL.speedTimer = setTimeout(() => { POL.speedTimer = null; polSpeedFlush(); }, 120);
}

function polSpeedFlush() {
  if (POL.speedPending === null) return;
  const v = POL.speedPending;
  POL.speedPending = null;
  api("/api/policy/command", { json: { speed: v } }).catch(() => { /* shown by api() */ });
}

/* Put the slider somewhere and send it now — the 0 button, and anything else that wants to
 * command a speed without a drag. */
function polSpeedSet(v) {
  const el = $("pol-speed");
  if (el.disabled) return;
  el.value = String(v);
  polSpeedInput();
  polSpeedFlush();
}

/* ================================================================ the older run/stop pair */

/* Is the robot being asked to travel? Not the same as the task flag: under the brake the flag
 * stays at 1 deliberately, and it is the schedule that is stopping the body. On a joystick run
 * the question is simply whether a nonzero speed is being asked for. */
function polGoing(p) {
  if (!p) return false;
  if (p.command_kind === "speed") return Math.abs(p.speed_cmd || 0) > 1e-3;
  return !!(p.run_flag && !p.braking);
}

/* The live command widget. Hidden entirely unless a run with a command channel is up, because a
 * control that cannot reach the policy is worse than no control: it does nothing while the
 * operator believes it did. Shown in ONE of its two shapes — slider or button pair. */
function polRenderCommand(st) {
  const p = st && st.policy;
  const box = $("pol-command");
  const live = !!(p && p.running && p.bundle_version === 2);
  box.classList.toggle("hidden", !live);
  if (!live) return;
  const joy = p.command_kind === "speed";
  $("pol-cmd-speed").classList.toggle("hidden", !joy);
  $("pol-cmd-flag").classList.toggle("hidden", joy);
  $("pol-command-blurb").innerHTML = joy
    ? `— the <b>speed</b> the policy is asked for, ${p.v_min.toFixed(2)} to ` +
      `${p.v_max.toFixed(2)} m/s, live, at full gains with the governor untouched. ` +
      "<b>0 is walk in place</b>, and it is the stop: this lineage was trained at every speed in " +
      "the range, so asking for 0 is an operating point and not a red light. Nothing here ends " +
      "the run — that is the End-run / Kill pair above."
    : "— commands to the <b>policy</b>, at full gains, with the governor untouched; neither one " +
      "ends the run. <b>RUN</b> hands the gait to the policy. <b>STOP</b> runs this checkpoint's " +
      "fitted brake schedule while <em>holding the task flag at 1</em> — telling the policy it " +
      "has finished is what makes it accelerate and fall (512/512 upright vs 3/512, measured " +
      "2026-09-11).";
  if (joy) {
    // a joystick trained with the RUN/STOP switch (RLframework stop_flag): task[1], live
    const sw = $("btn-pol-switch");
    sw.classList.toggle("hidden", !p.has_stop_switch);
    sw.disabled = p.phase !== "run";
    sw.textContent = p.run_flag ? "switch: RUN  (press for STOP)" : "switch: STOP  (press for RUN)";
    sw.onclick = () => api("/api/policy/command", { json: { run: !p.run_flag } }).catch(() => {});
    polRenderHeading(p); return polRenderSpeed(p);
  }
  polRenderHeading(p);
  const going = polGoing(p);
  const armed = p.phase === "run";                // before that the legs are still crawling
  box.classList.toggle("go", going);
  $("btn-pol-go").disabled = going || !armed;
  $("btn-pol-halt").disabled = !armed || p.braking || !p.has_brake;
  const state = $("pol-command-state");
  state.textContent = p.braking ? "BRAKING" : (going ? "RUNNING" : "HOLDING");
  state.className = "pol-cmdstate " + (p.braking ? "braking" : (going ? "on" : "off"));
  let note;
  if (!armed) {
    note = "waiting for the approach to reach the stance — the command goes live in the run phase";
  } else if (p.braking) {
    const pct = (p.brake_frac * 100).toFixed(0);
    note = `fitted brake schedule ${pct}% through its ${p.brake_window_s.toFixed(0)} s window ` +
           `· the task flag is deliberately still 1 · RUN hands the gait back to the policy`;
  } else if (going) {
    note = `running for ${p.run_flag_s.toFixed(1)} s · ${p.commits} gait cycles committed` +
           (p.has_brake ? "" : " · <b>no brake schedule in this bundle — STOP is unavailable</b>");
  } else if (!p.has_brake) {
    note = "<b>this bundle has no fitted brake schedule.</b> Fit one with " +
           "<code>walk_v2/tools/brake_search.py</code> and re-export with <code>--brake</code>; " +
           "dropping the task flag instead is measured at 3/512 upright.";
  } else {
    note = "holding the stance · RUN starts the gait";
  }
  $("pol-command-note").innerHTML = note;
}

/* The slider half. `speed_want` is where the operator put it, `speed_cmd` is what the control law
 * has slewed to and therefore what the policy actually read this tick, and `speed_est` is the
 * policy's own forward-speed estimate — the only speedometer on this robot, and the number that
 * says whether it is doing what it was asked. */
function polRenderSpeed(p) {
  const el = $("pol-speed");
  const armed = p.phase === "run";
  const key = `${p.file}|${p.v_min}|${p.v_max}`;
  if (POL.speedRun !== key) {                     // a new run: re-scale the track, park it at 0
    POL.speedRun = key;
    el.min = p.v_min;
    el.max = p.v_max;
    el.step = 0.05;
    el.value = p.speed_want || 0;
    POL.speedPending = null;
  }
  el.disabled = !armed;
  $("btn-pol-zero").disabled = !armed || (Math.abs(p.speed_want) < 1e-9 &&
                                          Math.abs(p.speed_cmd) < 1e-9);
  // never move the knob under a finger, or ahead of a send that has not landed yet
  if (!POL.speedDrag && POL.speedPending === null && document.activeElement !== el
      && Math.abs(+el.value - p.speed_want) > 1e-6) {
    el.value = p.speed_want;
  }
  const want = +el.value;
  $("pol-speed-val").textContent = `${want.toFixed(2)} m/s`;
  const ramping = Math.abs(p.speed_cmd - p.speed_want) > 0.01;
  const going = Math.abs(p.speed_cmd) > 1e-3;
  $("pol-command").classList.toggle("go", going);
  const state = $("pol-command-state");
  state.textContent = !armed ? "APPROACH"
                    : ramping ? (p.speed_cmd < p.speed_want ? "RAMPING UP" : "SLOWING")
                    : going ? "RUNNING" : "IN PLACE";
  state.className = "pol-cmdstate " + (!armed ? "off" : ramping ? "ramp" : going ? "on" : "off");
  // the command curriculum widens the draw band downward, so a mid-ramp checkpoint has never
  // been asked for the bottom of its own slider. Say so where the slider is, not only at arming.
  const tr = p.v_trained || [p.v_min, p.v_max];
  const untrained = want < tr[0] - 1e-6 || want > tr[1] + 1e-6;
  $("pol-command-note").innerHTML = !armed
    ? "waiting for the approach to reach the stance — the slider goes live in the run phase"
    : `applying <b>${p.speed_cmd.toFixed(2)}</b> m/s${ramping
        ? ` (ramping at ${p.cmd_slew.toFixed(2)} m/s²)` : ""} · policy's own estimate ` +
      `<b>${p.speed_est.toFixed(2)}</b> m/s · ${p.commits} gait cycles committed · ` +
      `${p.moving_s ? p.moving_s.toFixed(1) + " s asked to travel" : "not yet asked to travel"}` +
      (untrained
        ? `<br><b class="pf-bad">this checkpoint was only ever commanded ` +
          `${tr[0].toFixed(2)}–${tr[1].toFixed(2)} m/s</b> — ${want.toFixed(2)} is outside the ` +
          `band it was trained on`
        : "");
}

/* The heading half. This is the policy's OWN bearing estimate — an integrated gyro rate with an
 * origin, not a compass — and it is what the policy is steering on, so it is the number that
 * explains a veer. Zeroing it says "the direction it is pointing now is the one I want held";
 * nothing else moves the origin, deliberately, because a heading that quietly re-zeroed itself
 * would make the robot turn with no entry in the log to say why. */
function polRenderHeading(p) {
  const row = $("pol-cmd-heading");
  row.classList.toggle("hidden", !p.has_heading);
  if (!p.has_heading) return;
  const armed = p.phase === "run";
  const deg = p.heading_deg || 0;
  $("pol-heading-val").textContent = `${deg > 0 ? "+" : ""}${deg.toFixed(0)}°`;
  $("pol-heading-val").className = "pol-heading" + (Math.abs(deg) >= 15 ? " pf-bad"
                                                  : Math.abs(deg) >= 7 ? " warn" : "");
  $("btn-pol-aim").disabled = !armed;
  $("pol-heading-note").innerHTML = !armed
    ? "zeroed when the run reaches the stance — point the robot before you start it"
    : Math.abs(deg) >= 15
      ? "<b class=\"pf-bad\">the policy believes it has turned this far</b> — it is steering back " +
        "toward its own zero, which is not where you are standing"
      : "off its own straight-ahead by this much";
}

async function polZeroHeading() {
  try {
    await api("/api/policy/zero_heading", { json: {} });
    setBanner("straight ahead is now the direction the robot is pointing", "", 4000);
  } catch (e) {
    setBanner(e.message || "could not move the heading origin", "bad", 6000);
  }
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
  const cmdText = p.command_kind === "speed"
    ? `cmd ${p.speed_cmd.toFixed(2)} m/s` +
      (Math.abs(p.speed_cmd - p.speed_want) > 0.01 ? ` → ${p.speed_want.toFixed(2)}` : "")
    : p.bundle_version === 2
    ? `cmd ${p.braking ? "BRAKE " + (p.brake_frac * 100).toFixed(0) + "%"
                       : (p.run_flag ? "RUN" : "HOLD")}`
    : `cmd ${p.v_cmd.toFixed(2)} m/s · ${p.yaw_cmd.toFixed(2)} rad/s`;
  box.innerHTML =
    `<div class="row"><span class="pol-phase ${polEsc(p.phase)}">${polEsc(p.phase)}</span>` +
    `<b>${polEsc(p.run)}</b><span class="hint">${polEsc(p.file)}</span>` +
    `<span class="hint">${polEsc(cmdText)}</span></div>` +
    `<div class="pol-bar"><div style="width:${(frac * 100).toFixed(1)}%"></div></div>` +
    `<div class="ro-grid">` +
    `<div>elapsed</div><div>${p.elapsed_s.toFixed(2)} / ${p.max_seconds.toFixed(0)} s` +
      ` · ${p.ticks} ticks, ${p.late_ticks} late</div>` +
    `<div>gait</div><div>${p.gait_freq_hz.toFixed(2)} Hz, phase ${p.gait_phase.toFixed(2)}` +
      `${p.bundle_version === 2 ? ` · ${p.commits} spec commits` : ""}</div>` +
    `<div>rate</div><div>${p.rate_hz.toFixed(0)} / ${p.nominal_hz.toFixed(0)} Hz control` +
      `${p.decimation > 1 ? ` (every ${p.decimation}${p.decimation === 2 ? "nd" : "th"} loop tick)`
                          : ""} · ${p.step_ms.toFixed(1)} ms per tick</div>` +
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
  polRenderCommand(st);
  polRunButtons();
  if (polRunPhase() && !POL.deadman) polKeepaliveStart();   // adopt a run started in another tab
};

document.addEventListener("DOMContentLoaded", polInit);
