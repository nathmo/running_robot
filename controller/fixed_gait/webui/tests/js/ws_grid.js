/* Exercise the workspace editor's grid growth against the real app.js source.

   The grid is a safety object: every cell says "the leg may be commanded here". Growing the canvas
   renumbers every index, so a shift bug does not crash, it silently moves the safe region onto a
   different part of the joint. That is worth a harness. */
const fs = require("fs");
const path = require("path");

const SRC = fs.readFileSync(path.join(__dirname, "..", "..", "static", "app.js"), "utf8");

// Pull out just the pure grid functions, with their dependencies stubbed.
function extract(name) {
  const i = SRC.indexOf("function " + name + "(");
  if (i < 0) throw new Error("not found: " + name);
  let depth = 0, started = false;
  for (let k = i; k < SRC.length; k++) {
    if (SRC[k] === "{") { depth++; started = true; }
    else if (SRC[k] === "}") { depth--; if (started && depth === 0) return SRC.slice(i, k + 1); }
  }
  throw new Error("unterminated: " + name);
}

const harness = `
let wsEd = { grid: null, shape: null, camO: 0, thighO: 0, res: 1, dirty: false, world: [0,0] };
let S = { wsLeg: "left", ws: { limits: { left: {
  cam:   { hard: [-180, 180], nominal: [-88, 88] },
  thigh: { hard: [-180, 180], nominal: [-62, 62] },
} } } };
const HARD_FALLBACK = { abd: 48, cam: 88, thigh: 62 };
let BANNERS = [];
function setBanner(m) { BANNERS.push(m); }
function wsSyncExtentInputs() {}
function $(id) { return null; }
${extract("jointLimit")}
${extract("wsWorldToCell")}
${extract("wsGrow")}
${extract("wsGrowRoom")}
${extract("wsCellAt")}
${extract("wsFlagAtLimit")}
${extract("wsResize")}
module.exports = { get wsEd() { return wsEd; }, set wsEd(v) { wsEd = v; },
                   wsWorldToCell, wsGrow, wsGrowRoom, wsCellAt, wsResize,
                   get BANNERS() { return BANNERS; }, reset() { BANNERS = []; } };
`;

const M = new module.constructor();
M._compile(harness, "harness.js");
const H = M.exports;

let failures = 0;
function ok(cond, what) {
  if (cond) { console.log("  ok   " + what); }
  else { console.log("  FAIL " + what); failures++; }
}

function fresh(nc, nt, camO, thighO, res) {
  H.wsEd.grid = new Uint8Array(nc * nt);
  H.wsEd.shape = [nc, nt];
  H.wsEd.camO = camO; H.wsEd.thighO = thighO; H.wsEd.res = res || 1;
  H.wsEd.dirty = false;
  return H.wsEd;
}
function setCellWorld(camDeg, thDeg, v) {
  const [i, j] = H.wsWorldToCell(camDeg, thDeg);
  H.wsEd.grid[i * H.wsEd.shape[1] + j] = v === undefined ? 1 : v;
}
function getCellWorld(camDeg, thDeg) {
  const [i, j] = H.wsWorldToCell(camDeg, thDeg);
  const [nc, nt] = H.wsEd.shape;
  if (i < 0 || i >= nc || j < 0 || j >= nt) return null;
  return H.wsEd.grid[i * nt + j];
}
function occupied() {
  let n = 0;
  for (const v of H.wsEd.grid) n += v;
  return n;
}

console.log("\n1. growth preserves every painted cell AT ITS OWN ANGLE");
{
  // the reported geometry: a grid that stops at 36 deg of thigh
  fresh(79, 63, -38.6, -32.0, 1.0);
  setCellWorld(0.5, 30.5);
  setCellWorld(-30.5, -20.5);
  const before = occupied();
  H.wsGrow(40, 40, 50, 60);
  ok(occupied() === before, "cell count unchanged by a grow (" + before + ")");
  ok(getCellWorld(0.5, 30.5) === 1, "a painted cell is still painted at the same angle");
  ok(getCellWorld(-30.5, -20.5) === 1, "... and so is the other one");
  ok(getCellWorld(10.5, 10.5) === 0, "new area comes up empty");
  ok(H.wsEd.shape[0] === 159 && H.wsEd.shape[1] === 173, "shape grew as asked");
}

console.log("\n2. drawing past the edge grows the canvas (the reported bug)");
{
  fresh(79, 63, -38.6, -32.0, 1.0);          // thigh tops out at 31.0 deg
  const top = H.wsEd.thighO + H.wsEd.shape[1] * H.wsEd.res;
  ok(H.wsCellAt(0.0, top + 5.0, false) === null, "without grow, a stroke past the edge is dropped");
  const cell = H.wsCellAt(0.0, top + 5.0, true);
  ok(cell !== null, "with grow, the same point is reachable");
  const [nc, nt] = H.wsEd.shape;
  ok(H.wsEd.thighO + nt * H.wsEd.res > top, "the thigh span really did extend");
  ok(cell[1] === nt - 1, "the point lands in the newly added edge cell");
}

console.log("\n3. growth stops at the never-exceed limit and says so");
{
  fresh(79, 63, -38.6, -32.0, 1.0);
  H.reset();
  const cell = H.wsCellAt(0.0, 500.0, true);   // far past the +-180 thigh clamp
  const hi = H.wsEd.thighO + H.wsEd.shape[1] * H.wsEd.res;
  ok(hi <= 180.0 + 1e-9, "thigh span stopped at the clamp (" + hi.toFixed(1) + ")");
  ok(cell === null, "a point past the clamp is not reachable");
  ok(H.BANNERS.length > 0 && /never-exceed/.test(H.BANNERS[0]), "the operator is told why");
}

console.log("\n4. growth is bounded on every edge");
{
  fresh(10, 10, -5, -5, 1.0);
  H.wsCellAt(-1000, -1000, true);
  ok(H.wsEd.camO >= -180 - 1e-9, "cam low edge clamped (" + H.wsEd.camO.toFixed(1) + ")");
  ok(H.wsEd.thighO >= -180 - 1e-9, "thigh low edge clamped (" + H.wsEd.thighO.toFixed(1) + ")");
  const room = H.wsGrowRoom();
  ok(room.camLo === 0 && room.thLo === 0, "no room left on the grown edges");
}

console.log("\n5. sub-degree resolution survives a grow");
{
  fresh(40, 40, -10.0, -10.0, 0.5);
  setCellWorld(-3.75, 7.25);
  H.wsGrow(7, 3, 11, 5);
  ok(getCellWorld(-3.75, 7.25) === 1, "0.5 deg/cell grid keeps its cell at the same angle");
  ok(Math.abs(H.wsEd.camO - (-13.5)) < 1e-9, "origin moved by pad * res, not pad * 1");
}

console.log("\n6. resize can crop, and crops only what is outside");
{
  fresh(40, 40, -20.0, -20.0, 1.0);
  setCellWorld(-15.5, -15.5);      // will be cropped away
  setCellWorld(0.5, 0.5);          // will survive
  ok(occupied() === 2, "two cells painted");
  H.wsResize(-10, -10, -10, -10);  // crop 10 cells off each edge -> [-10, 10]
  ok(H.wsEd.shape[0] === 20 && H.wsEd.shape[1] === 20, "shape cropped");
  ok(Math.abs(H.wsEd.camO - (-10.0)) < 1e-9, "origin followed the crop");
  ok(occupied() === 1, "only the outside cell was discarded");
  ok(getCellWorld(0.5, 0.5) === 1, "the surviving cell is still at its own angle");
}

console.log("\n7. a stroke that grows mid-drag stays on the line (world coords, not indices)");
{
  fresh(79, 63, -38.6, -32.0, 1.0);
  // walk a diagonal that leaves the canvas partway, exactly as a drag would
  const pts = [];
  for (let t = 0; t <= 20; t++) pts.push([-5 + t * 1.0, 25 + t * 1.0]);
  let prevWorld = null;
  for (const [wx, wy] of pts) {
    const cell = H.wsCellAt(wx, wy, true);
    if (!cell) continue;
    if (prevWorld) {
      const prev = H.wsWorldToCell(...prevWorld);
      const steps = Math.max(Math.abs(cell[0] - prev[0]), Math.abs(cell[1] - prev[1]), 1);
      for (let s = 1; s <= steps; s++) {
        const i = Math.round(prev[0] + (cell[0] - prev[0]) * s / steps);
        const j = Math.round(prev[1] + (cell[1] - prev[1]) * s / steps);
        H.wsEd.grid[i * H.wsEd.shape[1] + j] = 1;
      }
    }
    H.wsEd.grid[cell[0] * H.wsEd.shape[1] + cell[1]] = 1;
    prevWorld = [wx, wy];
  }
  // every point of the drag must be painted, at its own angle
  let missing = [];
  for (const [wx, wy] of pts) if (getCellWorld(wx, wy) !== 1) missing.push([wx, wy]);
  ok(missing.length === 0, "every point of the drag is painted (" +
     (missing.length ? JSON.stringify(missing) : "none missing") + ")");
  // and nothing far off the line got painted
  ok(getCellWorld(-30.5, -25.5) === 0, "the brush did not smear across the shift");
}

console.log(failures ? `\n${failures} FAILURE(S)\n` : "\nall grid checks passed\n");
process.exit(failures ? 1 : 0);
