"""Read-only scanning of rl/runs/ — pure functions, no state, robust to legacy runs.

Facts to respect (see rl/train.py + SB3 logger behaviour):
- progress.csv header VARIES per run; the first data row may have blank train/* cells.
- time/fps is a CUMULATIVE average — never show it as current speed. Instantaneous
  sps = delta(time/total_timesteps) / delta(time/time_elapsed) between two rows.
- legacy runs may have only {"preset": name} in preset.json and no resolved_config.json.
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "rl" / "runs"

RUNNING_MTIME_S = 120          # progress.csv younger than this => training is alive
TAIL_BYTES = 64 * 1024         # enough for the last few CSV rows of any run


# ---------------------------------------------------------------- small readers
def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cell(s):
    """CSV cell -> float | None (blank/garbage -> None)."""
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_row(line, width):
    row = next(csv.reader([line]), [])
    vals = [_cell(c) for c in row]
    if width is not None:
        vals = (vals + [None] * width)[:width]
    return vals


def _tail_rows(csv_path, n=3):
    """(header list, last n data rows as float|None lists) — reads only head + tail bytes."""
    try:
        with open(csv_path, "rb") as f:
            header_line = f.readline().decode("utf-8", "replace").rstrip("\r\n")
            f.seek(0, 2)
            size = f.tell()
            off = max(len(header_line) + 1, size - TAIL_BYTES)
            f.seek(off)
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None, []
    if not header_line:
        return None, []
    header = next(csv.reader([header_line]), [])
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    if lines and off > len(header_line) + 1:
        lines = lines[1:]                 # first tail line may be a partial row
    if lines and lines[0] == header_line:
        lines = lines[1:]
    return header, [_parse_row(ln, len(header)) for ln in lines[-n:]]


def _col(header, rows, name):
    """Column values (None-padded) from the tail rows; [] if the column is absent."""
    try:
        i = header.index(name)
    except (ValueError, AttributeError):
        return []
    return [r[i] if i < len(r) else None for r in rows]


def _last(vals):
    for v in reversed(vals):
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------- public: scan
def scan_runs(managed_alive=None):
    """List every run under rl/runs/ with cheap live stats. managed_alive: {name: bool}
    from RunManager so freshly-launched runs show 'running' before the first CSV row."""
    managed_alive = managed_alive or {}
    out = []
    if not RUNS.is_dir():
        return out
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        out.append(_scan_one(d, managed_alive.get(d.name)))   # None = not managed
    out.sort(key=lambda r: -(r["mtimes"].get("dir") or 0))
    return out


def _scan_one(d, managed_alive):
    name = d.name
    src = _read_json(d / "preset.json") or {}
    desc = ""
    try:
        desc = (d / "description.txt").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    if not desc:
        desc = str(src.get("description") or "")
    if "preset" in src:
        source = {"kind": "preset", "base": src["preset"], "description": desc}
    elif "experiment" in src:
        source = {"kind": "experiment", "base": src["experiment"], "description": desc}
    elif "config" in src:
        source = {"kind": "config", "base": str(src["config"]), "description": desc}
    else:
        source = {"kind": "unknown", "base": None, "description": desc}

    resolved = _read_json(d / "resolved_config.json") or {}
    total = resolved.get("total_steps")
    n_envs = resolved.get("n_envs")

    prog = d / "progress.csv"
    prog_mtime = None
    steps = ep_rew = ep_len = sps = None
    if prog.exists():
        try:
            prog_mtime = prog.stat().st_mtime
        except OSError:
            pass
        header, rows = _tail_rows(prog, n=3)
        if header and rows:
            ts = _col(header, rows, "time/total_timesteps")
            el = _col(header, rows, "time/time_elapsed")
            steps = _last(ts)
            ep_rew = _last(_col(header, rows, "rollout/ep_rew_mean"))
            ep_len = _last(_col(header, rows, "rollout/ep_len_mean"))
            # instantaneous sps from the LAST TWO rows (delta-based; time/fps is cumulative)
            pairs = [(t, e) for t, e in zip(ts, el) if t is not None and e is not None]
            if len(pairs) >= 2:
                (t0, e0), (t1, e1) = pairs[-2], pairs[-1]
                if e1 > e0:
                    sps = (t1 - t0) / (e1 - e0)

    done = (d / "final_model.zip").exists()
    fresh = prog_mtime is not None and (time.time() - prog_mtime) < RUNNING_MTIME_S
    if managed_alive:                    # manager says the process is alive
        status = "running"
    elif done:
        status = "done"
    elif managed_alive is False:         # managed and DEAD (killed/crashed) — mtime lies
        status = "idle"
    elif fresh:                          # unmanaged: freshness heuristic
        status = "running"
    else:
        status = "idle"

    probe = _read_json(d / "gait_probe.json") or {}
    try:
        dir_mtime = max((prog_mtime or 0), d.stat().st_mtime)
    except OSError:
        dir_mtime = prog_mtime or 0

    return {
        "name": name,
        "source": source,
        "status": status,
        "managed": managed_alive,
        "steps": steps,
        "total": total,
        "n_envs": n_envs,
        "ep_rew": ep_rew,
        "ep_len": ep_len,
        "sps_inst": sps,
        "probe_verdict": probe.get("verdict"),
        "has_plots": (d / "training_plots.png").exists(),
        "has_video": (d / "run.mp4").exists() or (d / "eval.mp4").exists(),
        "artifacts": [f for f in ("training_plots.png", "run.mp4", "eval.mp4",
                                  "gait_probe.json") if (d / f).exists()],
        "mtimes": {"progress": prog_mtime, "dir": dir_mtime},
    }


# ---------------------------------------------------------------- public: progress
def read_progress(name, since_row=0, max_rows=4000):
    """Incremental progress.csv read by DATA-ROW index (header not counted).

    Re-reads the header every call (it is fixed per run but this stays correct if the
    file is rewritten). Blank cells -> null. Returns {header, rows, next, more}.
    """
    path = RUNS / name / "progress.csv"
    header, rows, more = [], [], False
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            width = len(header)
            idx = -1
            for raw in reader:
                if not any(c.strip() for c in raw):
                    continue
                idx += 1
                if idx < since_row:
                    continue
                if len(rows) >= max_rows:
                    more = True
                    break
                vals = [_cell(c) for c in raw]
                rows.append((vals + [None] * width)[:width])
    except OSError:
        pass
    return {"header": header, "rows": rows, "next": since_row + len(rows), "more": more}
