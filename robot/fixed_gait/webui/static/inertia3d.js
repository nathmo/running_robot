/* Self-contained WebGL viewer for the Limbs & Inertia panel: renders a limb STL mesh plus one or
 * two inertia ELLIPSOIDS (CAD vs identified) so a wrong reference frame is visible as a rotated
 * ellipsoid and wrong magnitudes as a differently-sized one. No external libraries, no CDN.
 * The ellipsoid is the uniform-density solid with the tensor's principal moments: its semi-axes and
 * orientation come straight from the eigen-decomposition the backend already computed. */
"use strict";

/* ---- tiny mat4/vec helpers (column-major, WebGL order) ---- */
const M4 = {
  mul(a, b) { const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s;
    } return o; },
  perspective(fov, asp, n, f) { const t = 1 / Math.tan(fov / 2); return new Float32Array(
    [t / asp, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0]); },
  lookAt(e, c, up) {
    const z = norm(sub(e, c)), x = norm(cross(up, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, e), -dot(y, e), -dot(z, e), 1]); },
  // model = translate(t) * rot3(R) * scale(s);  R is a flat 9 (row-major 3x3) or null
  model(t, R, s) {
    R = R || [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const m = new Float32Array(16);
    // columns are R's columns scaled; R row-major -> col j = [R[j], R[3+j], R[6+j]]
    for (let j = 0; j < 3; j++) { m[j * 4] = R[j] * s[j]; m[j * 4 + 1] = R[3 + j] * s[j];
      m[j * 4 + 2] = R[6 + j] * s[j]; m[j * 4 + 3] = 0; }
    m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1; return m; },
  rot3to4(R) { R = R || [1, 0, 0, 0, 1, 0, 0, 0, 1]; return new Float32Array(
    [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, 0, 0, 0, 1]); },
};
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

const VS = `attribute vec3 aPos; attribute vec3 aNormal;
uniform mat4 uMVP; uniform mat4 uNormal; varying vec3 vN;
void main(){ vN = normalize((uNormal*vec4(aNormal,0.0)).xyz); gl_Position = uMVP*vec4(aPos,1.0); }`;
const FS = `precision mediump float; varying vec3 vN;
uniform vec3 uColor; uniform float uAlpha; uniform vec3 uLight;
void main(){ float d = max(dot(normalize(vN), normalize(uLight)), 0.0);
  gl_FragColor = vec4(uColor*(0.35 + 0.65*d), uAlpha); }`;

function sphere(nu, nv) {                    // unit sphere positions+normals (normals==positions)
  const pos = [], idx = [];
  for (let i = 0; i <= nv; i++) { const th = i / nv * Math.PI;
    for (let j = 0; j <= nu; j++) { const ph = j / nu * 2 * Math.PI;
      pos.push(Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)); } }
  for (let i = 0; i < nv; i++) for (let j = 0; j < nu; j++) {
    const a = i * (nu + 1) + j, b = a + nu + 1;
    idx.push(a, b, a + 1, b, b + 1, a + 1); }
  return { pos: new Float32Array(pos), idx: new Uint16Array(idx) };
}

class Inertia3D {
  constructor(canvas) {
    this.cv = canvas;
    this.gl = canvas.getContext("webgl", { antialias: true, alpha: true });
    this.ok = !!this.gl;
    this.az = 0.7; this.el = 0.5; this.dist = 0.4; this.center = [0, 0, 0];
    this.mesh = null; this.ellipsoids = []; this.showMesh = true; this.align = null;
    if (!this.ok) return;
    const g = this.gl;
    this.prog = this._program(VS, FS);
    this.loc = {
      aPos: g.getAttribLocation(this.prog, "aPos"),
      aNormal: g.getAttribLocation(this.prog, "aNormal"),
      uMVP: g.getUniformLocation(this.prog, "uMVP"),
      uNormal: g.getUniformLocation(this.prog, "uNormal"),
      uColor: g.getUniformLocation(this.prog, "uColor"),
      uAlpha: g.getUniformLocation(this.prog, "uAlpha"),
      uLight: g.getUniformLocation(this.prog, "uLight"),
    };
    const s = sphere(28, 20);
    this.sphere = { pos: this._buf(s.pos), idx: this._ibuf(s.idx), n: s.idx.length };
    this._orbit();
    this._loop();
  }
  _program(vs, fs) { const g = this.gl;
    const c = (t, src) => { const sh = g.createShader(t); g.shaderSource(sh, src); g.compileShader(sh);
      if (!g.getShaderParameter(sh, g.COMPILE_STATUS)) console.error(g.getShaderInfoLog(sh)); return sh; };
    const p = g.createProgram(); g.attachShader(p, c(g.VERTEX_SHADER, vs));
    g.attachShader(p, c(g.FRAGMENT_SHADER, fs)); g.linkProgram(p); return p; }
  _buf(arr) { const g = this.gl, b = g.createBuffer(); g.bindBuffer(g.ARRAY_BUFFER, b);
    g.bufferData(g.ARRAY_BUFFER, arr, g.STATIC_DRAW); return b; }
  _ibuf(arr) { const g = this.gl, b = g.createBuffer(); g.bindBuffer(g.ELEMENT_ARRAY_BUFFER, b);
    g.bufferData(g.ELEMENT_ARRAY_BUFFER, arr, g.STATIC_DRAW); return b; }

  setMesh(arraybuffer, scale) {
    if (!this.ok) return;
    const dv = new DataView(arraybuffer);
    const n = dv.getUint32(80, true);
    if (80 + 4 + n * 50 > arraybuffer.byteLength) { this.mesh = null; return; } // not binary STL
    const pos = new Float32Array(n * 9), nrm = new Float32Array(n * 9);
    let o = 84, lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (let i = 0; i < n; i++) {
      const nx = dv.getFloat32(o, true), ny = dv.getFloat32(o + 4, true), nz = dv.getFloat32(o + 8, true);
      o += 12;
      for (let v = 0; v < 3; v++) {
        const x = dv.getFloat32(o, true) * scale, y = dv.getFloat32(o + 4, true) * scale,
          z = dv.getFloat32(o + 8, true) * scale; o += 12;
        const k = (i * 9) + v * 3;
        pos[k] = x; pos[k + 1] = y; pos[k + 2] = z;
        nrm[k] = nx; nrm[k + 1] = ny; nrm[k + 2] = nz;
        lo = [Math.min(lo[0], x), Math.min(lo[1], y), Math.min(lo[2], z)];
        hi = [Math.max(hi[0], x), Math.max(hi[1], y), Math.max(hi[2], z)];
      }
      o += 2;
    }
    this.mesh = { pos: this._buf(pos), nrm: this._buf(nrm), n: n * 3 };
    this.center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    this.dist = Math.max(Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 1.6, 0.05);
  }
  clearMesh() { this.mesh = null; }
  // ell: [{semi:[a,b,c], R:flat9(row-major), com:[x,y,z], color:[r,g,b]}]
  setEllipsoids(ell) { this.ellipsoids = ell || [];
    if (!this.mesh && this.ellipsoids.length) {
      const e = this.ellipsoids[0];
      this.center = e.com || [0, 0, 0];
      this.dist = Math.max(6 * Math.max(...(e.semi || [0.05])), 0.1);
    } }
  setOptions(o) { if (o.showMesh !== undefined) this.showMesh = o.showMesh;
    if (o.align !== undefined) this.align = o.align; }

  _orbit() {
    const cv = this.cv; let drag = null;
    cv.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
    cv.addEventListener("pointermove", (e) => { if (!drag) return;
      this.az += (e.clientX - drag[0]) * 0.01; this.el += (e.clientY - drag[1]) * 0.01;
      this.el = Math.max(-1.5, Math.min(1.5, this.el)); drag = [e.clientX, e.clientY]; });
    cv.addEventListener("pointerup", () => drag = null);
    cv.addEventListener("wheel", (e) => { e.preventDefault();
      this.dist *= e.deltaY < 0 ? 0.9 : 1.11; }, { passive: false });
  }
  _loop() { requestAnimationFrame(() => this._loop()); if (!this.ok || this.cv.offsetParent === null) return;
    this._render(); }
  _render() {
    const g = this.gl, w = this.cv.width, h = this.cv.height;
    g.viewport(0, 0, w, h); g.clearColor(0.06, 0.08, 0.11, 1);
    g.enable(g.DEPTH_TEST); g.clear(g.COLOR_BUFFER_BIT | g.DEPTH_BUFFER_BIT);
    g.useProgram(this.prog);
    const eye = [this.center[0] + this.dist * Math.cos(this.el) * Math.sin(this.az),
      this.center[1] + this.dist * Math.sin(this.el),
      this.center[2] + this.dist * Math.cos(this.el) * Math.cos(this.az)];
    const proj = M4.perspective(0.9, w / h, this.dist * 0.02, this.dist * 20);
    const view = M4.lookAt(eye, this.center, [0, 1, 0]);
    const VP = M4.mul(proj, view);
    g.uniform3fv(this.loc.uLight, [0.4, 0.7, 0.6]);
    if (this.mesh && this.showMesh) {
      this._drawBuf(this.mesh.pos, this.mesh.nrm, this.mesh.n, VP, M4.rot3to4(null),
        [0.55, 0.6, 0.68], 1.0, false);
    }
    g.enable(g.BLEND); g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA); g.depthMask(false);
    for (const e of this.ellipsoids) {
      let R = e.R;
      if (this.align && e.alignRot) R = e.alignRot;      // apply best-fit rotation to the CAD ellipsoid
      const model = M4.model(e.com || [0, 0, 0], R, e.semi);
      this._drawSphere(M4.mul(VP, model), M4.rot3to4(R), e.color, 0.32);
    }
    g.depthMask(true); g.disable(g.BLEND);
  }
  _bindAttr(posBuf, nrmBuf) { const g = this.gl;
    g.bindBuffer(g.ARRAY_BUFFER, posBuf); g.enableVertexAttribArray(this.loc.aPos);
    g.vertexAttribPointer(this.loc.aPos, 3, g.FLOAT, false, 0, 0);
    g.bindBuffer(g.ARRAY_BUFFER, nrmBuf); g.enableVertexAttribArray(this.loc.aNormal);
    g.vertexAttribPointer(this.loc.aNormal, 3, g.FLOAT, false, 0, 0); }
  _drawBuf(pos, nrm, count, mvp, nmat, color, alpha) { const g = this.gl;
    this._bindAttr(pos, nrm);
    g.uniformMatrix4fv(this.loc.uMVP, false, mvp); g.uniformMatrix4fv(this.loc.uNormal, false, nmat);
    g.uniform3fv(this.loc.uColor, color); g.uniform1f(this.loc.uAlpha, alpha);
    g.drawArrays(g.TRIANGLES, 0, count); }
  _drawSphere(mvp, nmat, color, alpha) { const g = this.gl;
    this._bindAttr(this.sphere.pos, this.sphere.pos);
    g.bindBuffer(g.ELEMENT_ARRAY_BUFFER, this.sphere.idx);
    g.uniformMatrix4fv(this.loc.uMVP, false, mvp); g.uniformMatrix4fv(this.loc.uNormal, false, nmat);
    g.uniform3fv(this.loc.uColor, color); g.uniform1f(this.loc.uAlpha, alpha);
    g.drawElements(g.TRIANGLES, this.sphere.n, g.UNSIGNED_SHORT, 0); }
}

window.Inertia3D = Inertia3D;
