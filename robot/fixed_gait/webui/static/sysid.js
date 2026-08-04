/* Dynamic-identification + Limbs & Inertia panels. Loaded after app.js, reuses its globals
 * (api, $, S, StripChart, setBanner, MOTORS). window.onSysidState(st) is called from app.js's
 * applyState (~2 Hz). All angles NORMALIZED degrees, same as the rest of the UI. */
"use strict";

const SID = {
  leg: "right", profile: "dynamic",
  curChart: null, model: null, selLimb: null, viewer: null,
  meshCache: {}, dynInit: false, runningWas: false,
  masses: null, massesKey: null,   // own copy: the limbs list must not depend on poll ordering
  measError: "",                   // last refused start/save, shown in-panel until the next attempt
};
const ROLES3 = ["abd", "cam", "thigh"];
const PROFILES = {
  "quasi_static": { f0: 0.03, f1: 0.03, amp: { abd: 8, cam: 12, thigh: 12 } },
  "dynamic": { f0: 0.05, f1: 0.6, amp: { abd: 6, cam: 9, thigh: 9 } },
};
const LIMB_COLOR = { cad: [0.30, 0.60, 1.0], identified: [0.95, 0.62, 0.13] };
const VERDICT_CLASS = { match: "v-ok", wrong_frame: "v-warn", wrong_values: "v-bad",
  incomplete: "v-none" };

/* ============================================================ measurement panel */
function buildMeasRows() {
  $("meas-joint-rows").innerHTML = ROLES3.map((r) => {
    const p = PROFILES[SID.profile];
    return `<tr data-role="${r}">
      <td>${r}</td>
      <td><input type="number" class="num small m-amp" value="${p.amp[r]}" step="1"></td>
      <td><input type="number" class="num small m-f0" value="${p.f0}" step="0.05"></td>
      <td><input type="number" class="num small m-f1" value="${p.f1}" step="0.05"></td></tr>`;
  }).join("");
}
function applyProfile(name) {
  SID.profile = name;
  const p = PROFILES[name];
  document.querySelectorAll("#meas-joint-rows tr").forEach((tr) => {
    const r = tr.dataset.role;
    tr.querySelector(".m-amp").value = p.amp[r];
    tr.querySelector(".m-f0").value = p.f0;
    tr.querySelector(".m-f1").value = p.f1;
  });
  $("meas-prof-dyn").classList.toggle("primary", name === "dynamic");
  $("meas-prof-static").classList.toggle("primary", name === "quasi_static");
  const base = SID.leg === "right" ? "measure_right" : "measure_left";
  $("meas-name").value = `${base}_${name === "dynamic" ? "dyn" : "static"}`;
}
function measSpec() {
  const amp = {}, f0 = {}, f1 = {};
  document.querySelectorAll("#meas-joint-rows tr").forEach((tr) => {
    const r = tr.dataset.role;
    amp[r] = +tr.querySelector(".m-amp").value;
    f0[r] = +tr.querySelector(".m-f0").value;
    f1[r] = +tr.querySelector(".m-f1").value;
  });
  return { leg: SID.leg, profile: SID.profile, duration: +$("meas-duration").value,
    ramp: +$("meas-ramp").value, hold_other: $("meas-holdother").checked,
    override: $("meas-override").checked, amp, f0, f1 };
}

function wireMeasure() {
  buildMeasRows();
  $("meas-tabs").querySelectorAll(".tab").forEach((b) => b.onclick = () => {
    $("meas-tabs").querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); SID.leg = b.dataset.leg; applyProfile(SID.profile);
  });
  $("meas-prof-static").onclick = () => applyProfile("quasi_static");
  $("meas-prof-dyn").onclick = () => applyProfile("dynamic");
  $("btn-meas-start").onclick = () => {
    measSetError("");
    api("/api/measure/start", { json: { spec: measSpec() } })
      .then(() => setBanner("measurement running…", "", 2500)).catch(measSetError);
  };
  $("btn-meas-stop").onclick = () =>
    api("/api/measure/stop", { method: "POST" }).catch(measSetError);
  $("btn-meas-finish").onclick = () =>
    api("/api/measure/finish", { json: { name: $("meas-name").value } })
      .then((d) => { if (d && d.ok) setBanner("saved " + (d.saved && d.saved.name), "", 3000); })
      .catch(measSetError);
  $("btn-identify-run").onclick = runIdentify;
  $("identify-import").onchange = importIdentify;
  SID.curChart = new StripChart($("meas-chart"), "#e0a020", 30);
}

async function runIdentify() {
  $("identify-status").textContent = "running identification…";
  try {
    const r = await fetch("/api/identify/run", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
    const d = await r.json();
    if (r.status === 501) {
      $("identify-status").innerHTML = "no mujoco/scipy on this host — run on the dev machine:<br>" +
        `<code>cd ${d.cwd} &amp;&amp; ${d.cli}</code><br>then import the JSON.`;
      return;
    }
    if (!d.ok) { $("identify-status").textContent = "failed: " + d.error; return; }
    $("identify-status").textContent = "identification complete ✓";
    setBanner("identification complete", "", 3000);
    loadModelInertia();
  } catch (e) { $("identify-status").textContent = "error: " + e; }
}
async function importIdentify(e) {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  const r = await fetch("/api/identify/import", { method: "POST", body: fd });
  const d = await r.json();
  setBanner(d.ok ? "identified params imported" : d.error, d.ok ? "" : "error", 4000);
  e.target.value = "";
  if (d.ok) loadModelInertia();
}

/** A refused start/save is the panel's own business: the global banner sits at the top of the page
 *  and this panel is at the bottom, so a rejection there reads as "the button does nothing".
 *  Held (not auto-hidden) until the next start attempt, and re-rendered on every poll below —
 *  updateMeasureUI owns #meas-status and would otherwise wipe it within 500 ms. */
function measSetError(e) {
  SID.measError = e ? (e.message || String(e)) : "";
  renderMeasStatus(S.state && S.state.measure);
}
function renderMeasStatus(m) {
  const el = $("meas-status");
  const running = !!(m && m.running);
  if (SID.measError && !running) {
    el.textContent = "⛔ " + SID.measError;
    el.classList.add("err");
    return;
  }
  el.classList.remove("err");
  el.textContent = m ? (running ? `running ${m.leg} ${m.profile} — ${m.elapsed}/${m.duration}s, ${m.n_samples} samples`
    : m.done ? `captured ${m.n_samples} samples — save it below` : "") : "";
}

function updateMeasureUI(st) {
  const m = st.measure;
  const running = !!(m && m.running);
  $("meas-phase").style.width = m ? (100 * (m.elapsed || 0) / (m.duration || 1)) + "%" : "0%";
  $("btn-meas-start").disabled = running;
  if (running) SID.measError = "";        // it started — the last rejection is history
  renderMeasStatus(m);
  // saved runs
  const runs = st.measurements || [];
  const box = $("meas-runs");
  const key = runs.map((r) => r.name).join("|");
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.innerHTML = runs.length ? runs.map((r) =>
      `<div class="meas-run"><b>${r.name}</b> <span class="hint">${r.leg || "?"} · ${r.profile || "?"} · ${r.n_samples || 0} samp · ${(r.duration_s || 0).toFixed(1)}s</span>
        <button class="btn small" data-exp="${r.name}">⬇</button>
        <button class="btn small" data-del="${r.name}">🗑</button></div>`).join("")
      : `<span class="hint">no runs captured yet</span>`;
    box.querySelectorAll("[data-exp]").forEach((b) => b.onclick = () =>
      window.location = "/api/measure/export?name=" + encodeURIComponent(b.dataset.exp));
    box.querySelectorAll("[data-del]").forEach((b) => b.onclick = () =>
      api("/api/measure/delete", { json: { name: b.dataset.del } }).catch(measSetError));
  }
}

/* live current chart of the excited leg (max |current| across its 3 joints) */
setInterval(() => {
  if (!SID.curChart || document.hidden) return;
  const st = S.state;
  if (st && st.measure && st.measure.running) {
    const leg = st.measure.leg;
    const vals = ROLES3.map((r) => (S.latest[leg + "." + r] || {}).cur).filter((v) => v != null);
    if (vals.length) SID.curChart.push((S.lastT || 0), Math.max(...vals.map(Math.abs)));
  }
  SID.curChart.draw(S.lastT || 0);
}, 120);

/* ============================================================ PID gains */
function buildPidRows(dyn) {
  const rows = MOTORS.map((m) => {
    const g = (dyn.pid || {})[m] || { kp: 0, ki: 0, kd: 0 };
    return `<tr data-motor="${m}"><td>${m}</td>
      <td><input type="number" class="num small p-kp" value="${g.kp}" step="0.01"></td>
      <td><input type="number" class="num small p-ki" value="${g.ki}" step="0.01"></td>
      <td><input type="number" class="num small p-kd" value="${g.kd}" step="0.001"></td></tr>`;
  }).join("");
  $("meas-pid-rows").innerHTML = rows;
  $("meas-pid-rows").querySelectorAll("tr").forEach((tr) => {
    const m = tr.dataset.motor;
    const send = () => api("/api/dynamics/pid", { json: { motor: m,
      kp: +tr.querySelector(".p-kp").value, ki: +tr.querySelector(".p-ki").value,
      kd: +tr.querySelector(".p-kd").value } }).catch(() => {});
    tr.querySelectorAll("input").forEach((i) => i.onchange = send);
  });
}

/* ============================================================ Limbs & inertia */
async function loadModelInertia() {
  try {
    SID.model = await (await fetch("/api/model/inertia")).json();
  } catch (e) { return; }
  // This runs at page load, BEFORE the first /api/state poll, so S.state is still empty here and
  // every mass field would paint blank. Fetch the masses ourselves for the first paint; after that
  // onSysidState keeps SID.masses current.
  if (SID.masses === null) {
    try {
      const st = await (await fetch("/api/state")).json();
      setLimbMasses(st.dynamics && st.dynamics.masses);
    } catch (e) { /* leave null; the next poll fills it in */ }
  }
  renderLimbsList();
  renderActuatorTable();
  if (SID.selLimb) selectLimb(SID.selLimb);
}

/** Store the polled masses; returns true if they changed (i.e. the list needs a repaint). */
function setLimbMasses(m) {
  const key = JSON.stringify(m || {});
  if (key === SID.massesKey) return false;
  SID.massesKey = key;
  SID.masses = m || {};
  return true;
}

function renderLimbsList() {
  const md = SID.model; if (!md || !md.bodies) return;
  const masses = SID.masses
    || (S.state && S.state.dynamics && S.state.dynamics.masses) || {};
  const names = Object.keys(md.bodies).filter((n) => !n.startsWith("motor_"));
  $("limbs-list").innerHTML = names.map((n) => {
    const cmp = md.bodies[n].comparison || {};
    const cat = cmp.category || "incomplete";
    const badge = { match: "matches", wrong_frame: "wrong frame", wrong_values: "off",
      incomplete: "—" }[cat];
    return `<div class="limb-row ${SID.selLimb === n ? "sel" : ""}" data-limb="${n}">
      <span class="limb-name">${n.replace("NCS-v1", "")}</span>
      <input type="number" class="num small limb-mass" data-body="${n}" placeholder="kg"
        value="${masses[n] != null ? masses[n] : ""}" step="0.001" title="weighed mass (kg)">
      <span class="verdict ${VERDICT_CLASS[cat]}">${badge}</span></div>`;
  }).join("");
  $("limbs-list").querySelectorAll(".limb-row").forEach((row) => {
    row.querySelector(".limb-name").onclick = () => selectLimb(row.dataset.limb);
    const mi = row.querySelector(".limb-mass");
    // The POST returns the updated config; take the masses straight from it, otherwise the repaint
    // below would briefly restore the pre-edit value (and, on a blank = revert-to-default, the
    // field would look like it did nothing until the next poll).
    mi.onchange = () => api("/api/dynamics/mass",
      { json: { body: mi.dataset.body, mass: mi.value === "" ? null : +mi.value } })
      .then((d) => { setLimbMasses(d && d.dynamics && d.dynamics.masses); loadModelInertia(); })
      .catch(() => {});
  });
  if (!SID.selLimb && names.length) selectLimb(names.find((n) => n.includes("Thigh")) || names[0]);
}

async function selectLimb(name) {
  SID.selLimb = name;
  document.querySelectorAll(".limb-row").forEach((r) =>
    r.classList.toggle("sel", r.dataset.limb === name));
  if (!SID.viewer && window.Inertia3D) SID.viewer = new Inertia3D($("limb-canvas"));
  const cmp = (SID.model.bodies[name] || {}).comparison || {};
  renderReadout(name, cmp);
  updateViewer(name, cmp);
  // load mesh (cached)
  if (SID.viewer && $("limb-showmesh").checked) {
    if (SID.meshCache[name] === undefined) {
      try {
        const buf = await (await fetch("/api/mesh/" + name + ".stl")).arrayBuffer();
        SID.meshCache[name] = buf;
      } catch (e) { SID.meshCache[name] = null; }
    }
    if (SID.meshCache[name]) SID.viewer.setMesh(SID.meshCache[name], 0.001);
    else SID.viewer.clearMesh();
    updateViewer(name, cmp);   // re-center on mesh
  }
}

function updateViewer(name, cmp) {
  if (!SID.viewer) return;
  const ell = [];
  const cad = cmp.cad, ident = cmp.identified;
  const flat = (M) => M ? [M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2]] : null;
  if (cad) ell.push({ semi: cad.ellipsoid_semi_axes, R: flat(cad.principal_axes),
    com: cad.com, color: LIMB_COLOR.cad, alignRot: cmp.rotation ? matMul3(flat(cmp.rotation.matrix), flat(cad.principal_axes)) : flat(cad.principal_axes) });
  if (ident) ell.push({ semi: ident.ellipsoid_semi_axes, R: flat(ident.principal_axes),
    com: ident.com, color: LIMB_COLOR.identified });
  SID.viewer.setEllipsoids(ell);
  SID.viewer.setOptions({ showMesh: $("limb-showmesh").checked, align: $("limb-align").checked });
}
function matMul3(A, B) {   // row-major 3x3 * 3x3
  const o = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++)
    o[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
  return o;
}

function renderReadout(name, cmp) {
  const el = $("limb-readout");
  if (!cmp || cmp.category === "incomplete") {
    el.innerHTML = `<b>${name}</b><br><span class="hint">${(cmp && cmp.verdict) || "no identified inertia yet — run identification"}</span>`;
    return;
  }
  const pm = cmp.principal_moments, r = cmp.rotation;
  const fmt6 = (a) => a.map((v) => v.toExponential(2)).join(", ");
  el.innerHTML = `<b>${name.replace("NCS-v1", "")}</b>
    <span class="verdict ${VERDICT_CLASS[cmp.category]}">${cmp.category.replace("_", " ")}</span>
    <p class="verdict-text">${cmp.verdict}</p>
    <div class="ro-grid">
      <div>principal moments CAD</div><div>${fmt6(pm.cad)}</div>
      <div>principal moments ident</div><div>${fmt6(pm.identified)}</div>
      <div>ratio (ident/CAD)</div><div>${pm.ratio.map((v) => v == null ? "—" : v.toFixed(2)).join(", ")}</div>
      <div>best-fit frame rotation</div><div>${r.angle_deg.toFixed(1)}°</div>
      <div>mass CAD → ident</div><div>${(cmp.mass.cad || 0).toFixed(3)} → ${(cmp.mass.identified || 0).toFixed(3)} kg</div>
      <div>suggested randomization</div><div>±${(cmp.suggested_dr_frac * 100).toFixed(0)}%</div>
    </div>`;
}

function renderActuatorTable() {
  const md = SID.model; if (!md) return;
  const val = (md.validation && md.validation.fit_residual_rms_nm) || {};
  $("actuator-rows").innerHTML = MOTORS.map((m) => {
    const kt = md.kt[m], arm = md.rotor_armature[m], fr = md.friction[m] || {};
    const g = (v, d = 3) => v == null ? "—" : (+v).toFixed(d);
    return `<tr><td>${m}</td><td>${g(kt)}</td><td>${g(arm, 4)}</td>
      <td>${g(fr.viscous)}</td><td>${g(fr.coulomb)}</td><td>${g(val[m], 2)}</td></tr>`;
  }).join("");
  const v = md.validation || {};
  $("identify-summary").innerHTML = md.has_identified
    ? `identified from ${(md.sources || []).join(", ") || "—"} · held-out normalized RMS ` +
      `<b>${v.heldout_normalized_rms != null ? (v.heldout_normalized_rms * 100).toFixed(0) + "%" : "—"}</b>` +
      ` (target ≤20%) · fit residual <b>${v.fit_residual_rms_nm_overall != null ? v.fit_residual_rms_nm_overall.toFixed(2) + " Nm" : "—"}</b>`
    : "no identified parameters yet — capture runs and click 'Identify', or import a JSON";
}

function wireLimbs() {
  $("limb-align").onchange = () => { if (SID.selLimb) updateViewer(SID.selLimb,
    (SID.model.bodies[SID.selLimb] || {}).comparison || {}); };
  $("limb-showmesh").onchange = () => { if (SID.selLimb) selectLimb(SID.selLimb); };
}

/* ============================================================ state hook + init */
window.onSysidState = function (st) {
  updateMeasureUI(st);
  if (st.dynamics && !SID.dynInit) { SID.dynInit = true; buildPidRows(st.dynamics); }
  // Repaint whenever the masses actually change (a default arriving, another tab's edit, a revert).
  // Cheap: setLimbMasses compares before anything is rebuilt, so a steady state costs one compare.
  if (st.dynamics && setLimbMasses(st.dynamics.masses) && SID.model) renderLimbsList();
  // refresh the inertia/actuator view once identification appears, or on first load
  if (st.identified !== undefined && SID.identifiedWas !== st.identified) {
    SID.identifiedWas = st.identified; loadModelInertia();
  }
};

(function initSysid() {
  wireMeasure();
  wireLimbs();
  applyProfile("dynamic");
  loadModelInertia();     // first paint (works before calibration — pure inspection)
})();
