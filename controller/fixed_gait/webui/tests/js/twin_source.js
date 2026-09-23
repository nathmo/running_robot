/* Which pose the digital twin draws, against the real twinAngles() in app.js.

   Three things can move the twin: the live drives, the gait preview, and — since 2026-09-23 — a
   policy DRY RUN, which is the one that moves nothing on the robot. Getting the order wrong is
   silent and dangerous in one direction: a twin drawing live telemetry during a rehearsal is a
   still picture of a limp robot, and a twin drawing a rehearsal when one is not running is a robot
   that appears to be walking. Neither throws, so it is worth a harness. */
const fs = require("fs");
const path = require("path");

const SRC = fs.readFileSync(path.join(__dirname, "..", "..", "static", "app.js"), "utf8");

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

const MOTOR_LINE = SRC.split("\n").find((l) => l.startsWith("const MOTORS ="));
if (!MOTOR_LINE) throw new Error("MOTORS not found in app.js");

const harness = `
${MOTOR_LINE}
const window = { policyTwinPose: null };
let S = { preview: { on: false }, traj: null, latest: {}, state: null };
function previewPhase() { return 0; }
function previewIdx(tr) { return 0; }
${extract("twinAngles")}
module.exports = { twinAngles, MOTORS, window,
                   get S() { return S; }, set S(v) { S = v; } };
`;

const M = new module.constructor();
M._compile(harness, "harness.js");
const H = M.exports;

let failures = 0;
function ok(cond, what) {
  if (cond) { console.log("  ok   " + what); }
  else { console.log("  FAIL " + what); failures++; }
}

const LIVE = { "right.abd": 1, "right.cam": 2, "right.thigh": 3,
               "left.abd": 4, "left.cam": 5, "left.thigh": 6 };
const DRY = { "right.abd": 11, "right.cam": 12, "right.thigh": 13,
              "left.abd": 14, "left.cam": 15, "left.thigh": 16 };

function setLive() {
  H.S.latest = {};
  for (const n of H.MOTORS) H.S.latest[n] = { pos_norm: LIVE[n] };
}
function setDry(pose) { H.window.policyTwinPose = pose === null ? null : () => pose; }

console.log("twin pose source");
setLive();
setDry(null);
let a = H.twinAngles();
ok(a.kind === "live" && a.norm["left.cam"] === 5, "with nothing else running, the twin is live");

setDry({ phase: "RUN", norm_deg: DRY });
a = H.twinAngles();
ok(a.kind === "dry" && a.norm["left.cam"] === 15,
   "a live dry run wins over the (motionless) telemetry underneath it");
ok(/RUN/.test(a.src), "the source line names the run's phase: " + a.src);

// the panel hands back null the moment the rehearsal ends or its pose goes stale; the twin must
// fall straight back to the robot rather than freeze on the last commanded stance
setDry(null);
a = H.twinAngles();
ok(a.kind === "live" && a.norm["left.cam"] === 5, "when the dry run ends, the twin is live again");

// the gait preview is an explicit toggle: someone holding it on is asking for that trajectory
H.S.preview = { on: true };
H.S.traj = { left: { path: [[21, 22]], abd_hold: 20 }, right: { path: [[31, 32]], abd_hold: 30 } };
setDry({ phase: "RUN", norm_deg: DRY });
a = H.twinAngles();
ok(a.kind === "preview" && a.norm["left.cam"] === 21,
   "the gait preview outranks a dry run");

H.S.preview = { on: false };
a = H.twinAngles();
ok(a.kind === "dry", "and the dry run comes back when the preview is switched off");

// a dry run that is missing a joint must draw that joint as absent, not as 0 quietly: null is what
// renderEE turns into "— " and a "no reading from" banner
setDry({ phase: "APPROACH", norm_deg: { "left.cam": 7 } });
a = H.twinAngles();
ok(a.norm["left.cam"] === 7 && a.norm["right.cam"] === null,
   "a pose missing a motor reports that motor as absent");

// an empty payload is not a pose at all
setDry({ phase: "RUN" });
a = H.twinAngles();
ok(a.kind === "live", "a record with no angles in it is not a pose");

if (failures) { console.log(failures + " FAILED"); process.exit(1); }
console.log("all twin source checks passed");
