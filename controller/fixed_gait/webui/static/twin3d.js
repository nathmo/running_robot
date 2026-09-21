/* Digital twin: renders the homing-pose MJCF (static/twin/, published by tools/build_twin.py) in
 * WebGL and poses it from the six motor angles. No external libraries, no CDN (the Pi has no
 * internet), and the MJCF is parsed HERE, in the browser — the page draws that file, not a copy.
 *
 * Joint map.  The drives are zeroed in the pose this MJCF was exported in, so every actuated joint
 * is simply   qpos = radians(sign * norm_deg)   with a sign per motor (twinmap.py, /api/twin/map).
 *
 * The four-bar.  The CAD export cannot express a closed loop, so it duplicates one body per leg
 * (left: the foot again under the pushrod; right: the pushrod again under the foot). The DEEPER copy
 * of a duplicated name is that closure copy: it is not drawn, and it only tells us which two bodies
 * the loop joins. Its joint is NOT the pin — the exporter copied the body's own joint onto it — so
 * the pin comes from the meshes: the pushrod's lower eye lies on its own axis, and the point on that
 * axis that meets the foot's 4 mm bore (foot frame -60.0, 33.0 mm) at qpos = 0 is 389.0 mm from the
 * upper pin with 0.00 mm lateral miss (fitted from the STLs, 2026-09-21; note the RL models' loop
 * site at -397.97 mm is the rod's outer EDGE, 9 mm further). The passive joints (pushrod on cam,
 * foot on thigh) are then solved each frame in closed form so both sides of the loop put the pin at
 * the same point, always on the assembly branch the CAD was exported in (see solveLoops).
 */
"use strict";
(() => {
  const TWIN_XML = "/static/twin/SpiderBotInitPos.xml";
  const PIN_BODY = /^Pushrod/;                  // the link whose pin we know
  const PIN_LOCAL = [0, 0, -0.389];             // lower eye centre, pushrod frame (m)
  const MOTOR_OF_BODY = [                        // which motor drives the joint INSIDE this body
    [/^HipLeft/, "left.abd"], [/^CamLeft/, "left.cam"], [/^ThighLeft/, "left.thigh"],
    [/^HipRight/, "right.abd"], [/^CamRight/, "right.cam"], [/^ThighRight/, "right.thigh"]];
  const COLOR_OF_BODY = [
    [/^body/, [0.55, 0.58, 0.64]], [/^Hip/, [0.40, 0.44, 0.50]], [/^Cam/, [0.88, 0.30, 0.30]],
    [/^Pushrod/, [0.88, 0.30, 0.30]], [/^Thigh/, [0.30, 0.64, 1.00]], [/^Foot/, [0.62, 0.80, 1.00]]];
  const LOOP_TOL = 2e-3;                         // m; a gap this big at the pin = not assemblable

  /* ---------------- 3x3 (row-major, flat 9) and vec3 ---------------- */
  const I3 = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const mm = (A, B) => { const o = new Array(9);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
      o[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
    return o; };
  const mv = (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
    A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];
  const mtv = (A, v) => [A[0] * v[0] + A[3] * v[1] + A[6] * v[2],
    A[1] * v[0] + A[4] * v[1] + A[7] * v[2], A[2] * v[0] + A[5] * v[1] + A[8] * v[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const nrm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dt = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const rotAxis = (u, q) => {                    // Rodrigues, unit axis u
    const c = Math.cos(q), s = Math.sin(q), t = 1 - c, [x, y, z] = u;
    return [t * x * x + c, t * x * y - s * z, t * x * z + s * y,
      t * x * y + s * z, t * y * y + c, t * y * z - s * x,
      t * x * z - s * y, t * y * z + s * x, t * z * z + c]; };
  const rx = (a) => rotAxis([1, 0, 0], a), ry = (a) => rotAxis([0, 1, 0], a), rz = (a) => rotAxis([0, 0, 1], a);
  const quatR = ([w, x, y, z]) => { const n = Math.hypot(w, x, y, z) || 1; w /= n; x /= n; y /= n; z /= n;
    return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]; };

  /* ---------------- MJCF ---------------- */
  const nums = (s, d) => (s == null ? d : s.trim().split(/\s+/).map(Number));

  function frameOf(el, ang, seq) {               // an element's local (R, p) from pos + euler/quat
    const p = nums(el.getAttribute("pos"), [0, 0, 0]);
    let R = I3();
    if (el.getAttribute("quat")) R = quatR(nums(el.getAttribute("quat")));
    else if (el.getAttribute("euler")) {
      // MuJoCo eulerseq: lowercase = axes that move with the frame (intrinsic) -> R = R1 R2 R3;
      // uppercase = fixed axes (extrinsic) -> R = R3 R2 R1. Default "xyz".
      const e = nums(el.getAttribute("euler")).map((v) => v * ang);
      const Rs = [...seq].map((ch, i) => ({ x: rx, y: ry, z: rz })[ch.toLowerCase()](e[i]));
      R = seq === seq.toLowerCase() ? mm(mm(Rs[0], Rs[1]), Rs[2]) : mm(mm(Rs[2], Rs[1]), Rs[0]);
    }
    return { R, p };
  }

  function parseMJCF(text) {
    const doc = new DOMParser().parseFromString(text, "application/xml");
    if (doc.querySelector("parsererror")) throw new Error("MJCF is not valid XML");
    const comp = doc.querySelector("compiler");
    // MuJoCo's default angle unit is DEGREE; this export says radian, but honour whatever it says
    const ang = comp && comp.getAttribute("angle") === "radian" ? 1 : Math.PI / 180;
    const seq = (comp && comp.getAttribute("eulerseq")) || "xyz";
    const meshdir = (comp && comp.getAttribute("meshdir")) || "";
    const meshes = {};
    for (const m of doc.querySelectorAll("asset > mesh"))
      meshes[m.getAttribute("name")] = { file: (meshdir ? meshdir.replace(/\/?$/, "/") : "") +
        m.getAttribute("file"), scale: nums(m.getAttribute("scale"), [1, 1, 1]) };
    const all = [];
    const walk = (el, parent, depth) => {
      const kids = [];
      for (const c of el.children) {
        if (c.tagName !== "body") continue;
        const f = frameOf(c, ang, seq);
        const j = [...c.children].find((x) => x.tagName === "joint" &&
          (x.getAttribute("type") || "hinge") === "hinge");
        const b = { name: c.getAttribute("name") || "", R0: f.R, p0: f.p, parent, depth, q: 0,
          joint: j ? { name: j.getAttribute("name"), axis: nrm(nums(j.getAttribute("axis"), [0, 0, 1])),
            pos: nums(j.getAttribute("pos"), [0, 0, 0]) } : null,
          geoms: [...c.children].filter((g) => g.tagName === "geom" && g.getAttribute("mesh"))
            .map((g) => ({ mesh: g.getAttribute("mesh"), ...frameOf(g, ang, seq) })),
          children: [], R: I3(), p: [0, 0, 0] };
        all.push(b);
        b.children = walk(c, b, depth + 1);
        kids.push(b);
      }
      return kids;
    };
    const roots = walk(doc.querySelector("worldbody"), null, 0);
    return { meshes, roots, all };
  }

  /* ---------------- kinematics ---------------- */
  function fk(roots) {
    const go = (b, Rp, pp) => {
      let R = mm(Rp, b.R0), p = add(pp, mv(Rp, b.p0));
      if (b.joint && b.q) {                      // hinge at joint.pos about joint.axis (body frame)
        const Rj = rotAxis(b.joint.axis, b.q);
        p = add(p, mv(R, sub(b.joint.pos, mv(Rj, b.joint.pos))));
        R = mm(R, Rj);
      }
      b.R = R; b.p = p;
      for (const c of b.children) go(c, R, p);
    };
    for (const r of roots) go(r, I3(), [0, 0, 0]);
  }
  const chain = (b) => { const out = []; for (; b; b = b.parent) out.push(b); return out; };

  /** Build the loops from duplicated body names. Needs fk() at qpos = 0 already run. */
  function findLoops(model) {
    const byName = new Map();
    for (const b of model.all) (byName.get(b.name) || byName.set(b.name, []).get(b.name)).push(b);
    const loops = [], closure = new Set();
    for (const [name, copies] of byName) {
      if (copies.length < 2) continue;
      copies.sort((a, b) => a.depth - b.depth);
      const primary = copies[0];
      for (const dup of copies.slice(1)) {
        const stack = [dup];                      // the closure copy and everything under it
        while (stack.length) { const x = stack.pop(); closure.add(x); stack.push(...x.children); }
        const a = dup.parent, b = primary;        // the loop joins these two drawn bodies
        const rod = PIN_BODY.test(a.name) ? a : PIN_BODY.test(b.name) ? b : null;
        if (!rod) { console.warn("twin: loop " + name + " has no pushrod; left open"); continue; }
        const other = rod === a ? b : a;
        const inA = new Set(chain(rod)), inB = new Set(chain(other));
        const passive = [...chain(rod), ...chain(other)].filter((x) =>
          x.joint && !x.motor && !(inA.has(x) && inB.has(x)));
        // the closed form below needs exactly this: a planar four-bar whose two passive pins are
        // the rod's own joint and the other body's own joint. Refuse anything else loudly.
        if (passive.length !== 2 || !passive.includes(rod) || !passive.includes(other))
          throw new Error(`loop ${name}: passive joints [${passive.map((x) => x.name)}] are not ` +
            `the pins of ${rod.name} and ${other.name} — not the four-bar this viewer solves`);
        const pinW = add(rod.p, mv(rod.R, PIN_LOCAL));             // at qpos = 0
        const L = { name, rod, other, pinRod: PIN_LOCAL, pinOther: mtv(other.R, sub(pinW, other.p)),
          resid: 0 };
        const g = loopGeom(L);
        L.Lp = Math.hypot(...g.flat(sub(g.A0, g.P)));             // pin-to-pin, rod
        L.La = Math.hypot(...g.flat(sub(g.B0, g.K)));             // knee pin to rod pin, other body
        // which of the two assemblies the CAD is in: side of the P->K line the pin sits on
        L.branch = Math.sign(dt(crs(g.n, g.flat(sub(g.K, g.P))), g.flat(sub(g.A0, g.P)))) || 1;
        loops.push(L);
      }
    }
    for (const x of closure) {                   // detach closure copies from the drawn tree
      const sib = x.parent && x.parent.children;
      if (sib && sib.includes(x)) sib.splice(sib.indexOf(x), 1);
    }
    model.all = model.all.filter((x) => !closure.has(x));
    return loops;
  }

  function loopResidual(L) {
    return sub(add(L.rod.p, mv(L.rod.R, L.pinRod)), add(L.other.p, mv(L.other.R, L.pinOther)));
  }

  /** World geometry of a loop with its two passive pins at 0 (fk() must have run that way):
   *  the rod pivot P, the other body's pivot K, the pin as carried by each (A0, B0), the plane
   *  normal n (the rod's joint axis) and a projector onto that plane. */
  function loopGeom(L) {
    const { rod, other } = L;
    const n = nrm(mv(rod.R, rod.joint.axis));
    return { n, flat: (v) => sub(v, n.map((x) => x * dt(v, n))),
      P: add(rod.p, mv(rod.R, rod.joint.pos)), K: add(other.p, mv(other.R, other.joint.pos)),
      A0: add(rod.p, mv(rod.R, L.pinRod)), B0: add(other.p, mv(other.R, L.pinOther)) };
  }
  const angleAbout = (ax, v0, v1) => Math.atan2(dt(ax, crs(v0, v1)), dt(v0, v1));

  /** Closed form, circle-circle in the loop's plane: the pin is Lp from P and La from K, on the
   *  side of P->K the CAD was assembled on (L.branch). A numeric solver cannot hold that side: at
   *  the homing pose this loop is 5 mm short of full stretch (pushrod and knee stub 155 deg apart),
   *  and a Gauss-Newton / LM solve that is pushed through the stretch — a zero off by a degree or
   *  two does it — comes back out on the OTHER assembly and stays there (seen in testing). Past the
   *  stretch (no assembly exists) the pin is put on the P->K line, i.e. the closest pose, and the
   *  residual reports the gap. */
  function solveLoops(model) {
    for (const L of model.loops) {
      L.rod.q = 0; L.other.q = 0;
    }
    fk(model.roots);
    for (const L of model.loops) {
      const g = loopGeom(L), d = g.flat(sub(g.K, g.P)), D = Math.hypot(...d);
      if (D < 1e-9) continue;
      const u = d.map((x) => x / D), w = crs(g.n, u);
      const a = (L.Lp * L.Lp - L.La * L.La + D * D) / (2 * D);
      const h = Math.sqrt(Math.max(L.Lp * L.Lp - a * a, 0));
      const X = add(g.P, add(u.map((x) => x * a), w.map((x) => x * h * L.branch)));
      L.rod.q = angleAbout(g.n, g.flat(sub(g.A0, g.P)), g.flat(sub(X, g.P)));
      L.other.q = angleAbout(nrm(mv(L.other.R, L.other.joint.axis)),
        g.flat(sub(g.B0, g.K)), g.flat(sub(X, g.K)));
    }
    fk(model.roots);
    for (const L of model.loops) L.resid = Math.hypot(...loopResidual(L));
  }

  /* ---------------- STL ---------------- */
  function parseSTL(buf, scale) {
    const dv = new DataView(buf), n = dv.getUint32(80, true);
    if (84 + n * 50 !== buf.byteLength) throw new Error("not a binary STL");
    const pos = new Float32Array(n * 9), nor = new Float32Array(n * 9);
    let o = 84;
    for (let i = 0; i < n; i++) {
      o += 12;                                    // file normal ignored: recomputed from winding
      const v = [];
      for (let k = 0; k < 3; k++) { v.push([dv.getFloat32(o, true) * scale[0],
        dv.getFloat32(o + 4, true) * scale[1], dv.getFloat32(o + 8, true) * scale[2]]); o += 12; }
      o += 2;
      const fn = nrm(crs(sub(v[1], v[0]), sub(v[2], v[0])));
      for (let k = 0; k < 3; k++) { pos.set(v[k], i * 9 + k * 3); nor.set(fn, i * 9 + k * 3); }
    }
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < pos.length; i += 3) for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], pos[i + k]); hi[k] = Math.max(hi[k], pos[i + k]); }
    return { pos, nor, n: n * 3, lo, hi };
  }

  /* ---------------- GL ---------------- */
  const VS = `attribute vec3 aPos; attribute vec3 aNor;
uniform mat4 uVP; uniform mat4 uM; varying vec3 vN;
void main(){ vN = (uM*vec4(aNor,0.0)).xyz; gl_Position = uVP*uM*vec4(aPos,1.0); }`;
  const FS = `precision mediump float; varying vec3 vN; uniform vec3 uColor; uniform vec3 uLight;
uniform vec3 uView; uniform float uFlat;
void main(){ vec3 n = normalize(vN);
  float d = abs(dot(n, uLight)); float h = abs(dot(n, uView));
  vec3 c = uFlat > 0.5 ? uColor : uColor*(0.30 + 0.45*d + 0.30*h);
  gl_FragColor = vec4(c, 1.0); }`;
  const m4 = (R, p) => new Float32Array([R[0], R[3], R[6], 0, R[1], R[4], R[7], 0,
    R[2], R[5], R[8], 0, p[0], p[1], p[2], 1]);
  const m4mul = (a, b) => { const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; }
    return o; };
  const persp = (fov, asp, n, f) => { const t = 1 / Math.tan(fov / 2); return new Float32Array(
    [t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0]); };
  const lookAt = (e, c, up) => { const z = nrm(sub(e, c)), x = nrm(crs(up, z)), y = crs(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dt(x, e), -dt(y, e), -dt(z, e), 1]); };

  const VIEWS = {                                 // az about +z from +x (forward), el up
    left: [Math.PI / 2, 0.05], right: [-Math.PI / 2, 0.05], front: [0, 0.1], iso: [0.75, 0.35] };

  class Twin3D {
    constructor(canvas) {
      this.cv = canvas;
      this.gl = canvas.getContext("webgl", { antialias: true });
      this.ok = !!this.gl;
      this.model = null; this.error = null; this.dirty = true;
      [this.az, this.el] = VIEWS.iso; this.dist = 1.9; this.center = [0, 0, -0.33];
      this.badSides = [];
      if (!this.ok) { this.error = "WebGL unavailable in this browser"; return; }
      const g = this.gl;
      const sh = (t, s) => { const x = g.createShader(t); g.shaderSource(x, s); g.compileShader(x);
        if (!g.getShaderParameter(x, g.COMPILE_STATUS)) console.error(g.getShaderInfoLog(x)); return x; };
      this.prog = g.createProgram();
      g.attachShader(this.prog, sh(g.VERTEX_SHADER, VS)); g.attachShader(this.prog, sh(g.FRAGMENT_SHADER, FS));
      g.linkProgram(this.prog);
      const U = (n) => g.getUniformLocation(this.prog, n);
      this.loc = { aPos: g.getAttribLocation(this.prog, "aPos"), aNor: g.getAttribLocation(this.prog, "aNor"),
        uVP: U("uVP"), uM: U("uM"), uColor: U("uColor"), uLight: U("uLight"), uView: U("uView"),
        uFlat: U("uFlat") };
      this._axes();
      this._orbit();
      this._loop();
    }

    async load(url = TWIN_XML) {
      if (!this.ok) return;
      try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`${url}: HTTP ${r.status} (run tools/build_twin.py and copy static/twin/)`);
        const model = parseMJCF(await r.text());
        for (const b of model.all) {
          const hit = MOTOR_OF_BODY.find(([re]) => re.test(b.name));
          b.motor = hit && b.joint ? hit[1] : null;
          const col = COLOR_OF_BODY.find(([re]) => re.test(b.name));
          b.color = col ? col[1] : [0.7, 0.7, 0.7];
        }
        fk(model.roots);
        model.loops = findLoops(model);
        model.byMotor = {};
        for (const b of model.all) if (b.motor) model.byMotor[b.motor] = b;
        const base = url.slice(0, url.lastIndexOf("/") + 1), g = this.gl;
        const need = new Set(model.all.flatMap((b) => b.geoms.map((x) => x.mesh)));
        model.gl = {};
        await Promise.all([...need].map(async (name) => {
          const m = model.meshes[name];
          if (!m) return;
          const rr = await fetch(base + m.file);
          if (!rr.ok) throw new Error(`${m.file}: HTTP ${rr.status}`);
          const s = parseSTL(await rr.arrayBuffer(), m.scale);
          const buf = (a) => { const b = g.createBuffer(); g.bindBuffer(g.ARRAY_BUFFER, b);
            g.bufferData(g.ARRAY_BUFFER, a, g.STATIC_DRAW); return b; };
          model.gl[name] = { pos: buf(s.pos), nor: buf(s.nor), n: s.n, lo: s.lo, hi: s.hi };
        }));
        this._frame(model);
        this.model = model;
        this.error = null;
        this.dirty = true;
      } catch (e) {
        this.error = "twin model: " + e.message;
        console.error(e);
      }
    }

    /** angles: {motor: qpos rad | null}. Returns per-leg loop residuals (m). */
    setPose(qByMotor) {
      const m = this.model;
      if (!m) return null;
      for (const [motor, b] of Object.entries(m.byMotor)) {
        const q = qByMotor[motor];
        b.q = Number.isFinite(q) ? q : 0;
      }
      solveLoops(m);
      // tint only the leg whose loop is open (loop names carry Left/Right, as every leg body does)
      this.badSides = m.loops.filter((L) => L.resid >= LOOP_TOL)
        .map((L) => (/Left/.exec(L.name) || /Right/.exec(L.name) || [""])[0]).filter(Boolean);
      this.dirty = true;
      return m.loops.map((L) => ({ name: L.name, resid: L.resid }));
    }

    view(name) {
      if (!VIEWS[name]) return;
      [this.az, this.el] = VIEWS[name];
      if (this.home) { this.center = this.home.center; this.dist = this.home.dist; }
      this.dirty = true;
    }

    /** Aim the orbit camera at the whole robot as loaded (qpos = 0): the world bounding box of
     *  every mesh's own box corners. The view buttons return to this framing. */
    _frame(model) {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const b of model.all) for (const geo of b.geoms) {
        const m = model.gl[geo.mesh];
        if (!m) continue;
        const R = mm(b.R, geo.R), p = add(b.p, mv(b.R, geo.p));
        for (let c = 0; c < 8; c++) {
          const w = add(p, mv(R, [c & 1 ? m.hi[0] : m.lo[0], c & 2 ? m.hi[1] : m.lo[1], c & 4 ? m.hi[2] : m.lo[2]]));
          for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], w[k]); hi[k] = Math.max(hi[k], w[k]); }
        }
      }
      if (!Number.isFinite(lo[0])) return;
      this.center = [0, 1, 2].map((k) => (lo[k] + hi[k]) / 2);
      // radius of the box; fov 0.55 rad -> a sphere of radius r fills the height at r / sin(0.275)
      const r = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
      this.dist = 1.15 * r / Math.sin(0.275);
      this.home = { center: this.center, dist: this.dist };
      this.dirty = true;
    }

    _axes() {                                     // world triad at the torso origin, 10 cm
      const g = this.gl, P = [], N = [];
      for (let k = 0; k < 3; k++) { const e = [0, 0, 0]; e[k] = 0.1; P.push(0, 0, 0, ...e); N.push(0, 0, 1, 0, 0, 1); }
      const b = (a) => { const x = g.createBuffer(); g.bindBuffer(g.ARRAY_BUFFER, x);
        g.bufferData(g.ARRAY_BUFFER, new Float32Array(a), g.STATIC_DRAW); return x; };
      this.axes = { pos: b(P), nor: b(N) };
    }
    _orbit() {
      const cv = this.cv; let drag = null;
      cv.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
      cv.addEventListener("pointermove", (e) => { if (!drag) return;
        this.az -= (e.clientX - drag[0]) * 0.01; this.el += (e.clientY - drag[1]) * 0.01;
        this.el = Math.max(-1.5, Math.min(1.5, this.el)); drag = [e.clientX, e.clientY]; this.dirty = true; });
      cv.addEventListener("pointerup", () => { drag = null; });
      cv.addEventListener("wheel", (e) => { e.preventDefault();
        this.dist *= e.deltaY < 0 ? 0.9 : 1.11; this.dirty = true; }, { passive: false });
    }
    _loop() {
      requestAnimationFrame(() => this._loop());
      if (this.cv.offsetParent === null) return;             // panel hidden: draw nothing
      const w = Math.round(this.cv.clientWidth * (window.devicePixelRatio || 1));
      const h = Math.round(this.cv.clientHeight * (window.devicePixelRatio || 1));
      if (w && h && (this.cv.width !== w || this.cv.height !== h)) { this.cv.width = w; this.cv.height = h; this.dirty = true; }
      if (!this.dirty) return;
      this.dirty = false;
      this._render();
    }
    _render() {
      const g = this.gl, w = this.cv.width, h = this.cv.height, L = this.loc;
      g.viewport(0, 0, w, h); g.clearColor(0.055, 0.078, 0.106, 1);
      g.enable(g.DEPTH_TEST); g.clear(g.COLOR_BUFFER_BIT | g.DEPTH_BUFFER_BIT);
      if (!this.model) return;
      g.useProgram(this.prog);
      const c = this.center, ce = Math.cos(this.el);
      const eye = [c[0] + this.dist * ce * Math.cos(this.az), c[1] + this.dist * ce * Math.sin(this.az),
        c[2] + this.dist * Math.sin(this.el)];
      const VP = m4mul(persp(0.55, w / h, this.dist * 0.05, this.dist * 10), lookAt(eye, c, [0, 0, 1]));
      g.uniformMatrix4fv(L.uVP, false, VP);
      g.uniform3fv(L.uLight, nrm([0.3, 0.5, 0.8]));
      g.uniform3fv(L.uView, nrm(sub(eye, c)));
      const bind = (pos, nor) => {
        g.bindBuffer(g.ARRAY_BUFFER, pos); g.enableVertexAttribArray(L.aPos);
        g.vertexAttribPointer(L.aPos, 3, g.FLOAT, false, 0, 0);
        g.bindBuffer(g.ARRAY_BUFFER, nor); g.enableVertexAttribArray(L.aNor);
        g.vertexAttribPointer(L.aNor, 3, g.FLOAT, false, 0, 0); };
      g.uniform1f(L.uFlat, 0);
      for (const b of this.model.all) {
        const legBad = this.badSides.some((s) => b.name.includes(s));
        g.uniform3fv(L.uColor, legBad ? [0.9, 0.6, 0.15] : b.color);
        for (const geo of b.geoms) {
          const buf = this.model.gl[geo.mesh];
          if (!buf) continue;
          g.uniformMatrix4fv(L.uM, false, m4(mm(b.R, geo.R), add(b.p, mv(b.R, geo.p))));
          bind(buf.pos, buf.nor);
          g.drawArrays(g.TRIANGLES, 0, buf.n);
        }
      }
      // world triad drawn on top: x red (forward), y green (left), z blue (up)
      g.disable(g.DEPTH_TEST); g.uniform1f(L.uFlat, 1);
      g.uniformMatrix4fv(L.uM, false, m4(I3(), [0, 0, 0]));
      bind(this.axes.pos, this.axes.nor);
      [[1, 0.3, 0.3], [0.3, 1, 0.3], [0.4, 0.6, 1]].forEach((col, k) => {
        g.uniform3fv(L.uColor, col); g.drawArrays(g.LINES, k * 2, 2); });
    }
  }

  Twin3D.LOOP_TOL = LOOP_TOL;
  window.Twin3D = Twin3D;
  // exported for tests / console poking: the kinematics without the GL
  window.Twin3DKin = { parseMJCF, fk, findLoops, solveLoops, loopResidual, parseSTL };
})();
