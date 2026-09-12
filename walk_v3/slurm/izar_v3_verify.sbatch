#!/bin/bash -l
# The v3 verification suite, run unattended after training.
#
# Submit it with a dependency so it fires the moment the training jobs stop, however they stop:
#   sbatch --dependency=afterany:<jobid>[:<jobid>...] \
#          --export=ALL,RUN=walk_v3/runs/v3_joy_s7 walk_v3/slurm/izar_v3_verify.sbatch
#
# afterany, not afterok: a training job that dies at the wall clock still leaves checkpoints worth
# measuring, and a suite that silently never runs is worse than one that reports on a partial run.
#
# Everything here is read-only with respect to the run directory except results/, so re-running is safe.
#SBATCH --job-name=v3-verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=walk_v3/logs/%x-%j.out

set -uo pipefail          # NOT -e: one failing probe must not cancel the rest of the suite
module purge
export PATH=$HOME/.local/bin:$PATH
source "${VENV:-$HOME/venvs/dash-v2}/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
cd "${REPO:-$HOME/running_robot}"

RUN="${RUN:?set RUN=walk_v3/runs/<name>}"
OUT="${OUT:-walk_v3/results/$(basename "$RUN")}"
mkdir -p "$OUT"

# Prefer the keeper (best by survivors, then command tracking); fall back to the newest checkpoint.
CK="$RUN/best.msgpack"
[ -f "$CK" ] || CK=$({ ls -1t "$RUN"/final.msgpack "$RUN"/ckpt_*.msgpack 2>/dev/null || true; } | head -1)
if [ -z "${CK:-}" ] || [ ! -f "$CK" ]; then
    echo "[verify] FATAL: no checkpoint under $RUN" >&2; exit 1
fi
echo "[verify] run $RUN"
echo "[verify] checkpoint $CK"
echo "[verify] results -> $OUT"

step () { echo; echo "################ $* ################"; }

step "1. command sweep, nominal start -- THE deliverable: does the stick mean what it says?"
python walk_v3/tools/command_sweep.py --run "$RUN" --checkpoint "$CK" \
    --json "$OUT/command_sweep.json" 2>&1 | grep -vE "Failed to import (mujoco_)?warp"

step "2. command sweep, full bring-up -- the same stick after a drop or a tilted release"
python walk_v3/tools/command_sweep.py --run "$RUN" --checkpoint "$CK" --bringup full \
    --json "$OUT/command_sweep_bringup.json" 2>&1 | grep -vE "Failed to import (mujoco_)?warp"

step "3. command sweep on a randomised plant -- does tracking survive the DR draw?"
python walk_v3/tools/command_sweep.py --run "$RUN" --checkpoint "$CK" --dr \
    --json "$OUT/command_sweep_dr.json" 2>&1 | grep -vE "Failed to import (mujoco_)?warp"

step "4. no-privilege audit -- the actor must be a pure function of measurable inputs"
python walk_v3/tools/audit_actor_obs.py --run "$RUN" --checkpoint "$CK" 2>&1 \
    | grep -vE "Failed to import (mujoco_)?warp"
echo "[verify] audit exit ${PIPESTATUS[0]} (0 = clean; this is the claim that v2 checkpoints FAIL)"

step "5. reward budget at the settled policy -- is living still profitable?"
for V in 3.2 1.8 0.0; do
    python walk_v3/tools/reward_budget.py --run "$RUN" --checkpoint "$CK" --v-cmd "$V" \
        --curriculum final --raw-stats 2>&1 | grep -vE "Failed to import (mujoco_)?warp"
done

step "6. bring-up envelope -- hold, tilt and handover against the measured +-5 deg"
# --sweep takes ONE mode, not a comma list, so run the three that matter separately.
for MODE in hold tilt handover; do
    echo "---- bring-up sweep: $MODE ----"
    python walk_v3/tools/bringup_probe.py --run "$RUN" --checkpoint "$CK" --sweep "$MODE" 2>&1         | grep -vE "Failed to import (mujoco_)?warp"
done

step "7. deploy: export the bundle, then the offline dress rehearsal"
python walk_v3/export.py --run "$RUN" --checkpoint "$CK" --out "$OUT/bundle" 2>&1     | grep -vE "Failed to import (mujoco_)?warp"
# --mock is the no-hardware rehearsal (mock bus, mock IMU); the real preflight in run_policy.py needs
# CAN buses and cannot run here. A speed command exercises the v3 path, so a joystick bundle with a
# wrong task[1] or a missing v_max fails HERE rather than on the robot.
python robot/deploy/run_policy.py --bundle "$OUT/bundle" --mock --max-seconds 5 --speed 1.5 2>&1     | tail -n 40

echo
echo "[verify] SUITE COMPLETE for $RUN -- results under $OUT"
