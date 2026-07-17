"""Flask routes for the run orchestrator. Handlers only read scanner/RunManager
snapshots and post requests — the RunManager daemon owns the subprocesses."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .runmanager import ROOT, RUNS, NAME_RE, RunManager, TOOL_KINDS
from . import scanner

app = Flask(__name__, static_folder="static", static_url_path="/static")
MANAGER = RunManager()

FILE_WHITELIST = {"training_plots.png", "run.mp4", "eval.mp4", "gait_probe.json"}
LOG_CHUNK = 64 * 1024


def _ok(**extra):
    return jsonify({"ok": True, **extra})


def _err(msg, code=400):
    return jsonify({"ok": False, "error": str(msg)}), code


def _check(name):
    return bool(NAME_RE.match(name or ""))


# ================================================================ experiments (10 s cache)
_EXP_CACHE = {"t": 0.0, "data": None}
_YAML_KEY = re.compile(r"^(name|description):\s*(.*?)\s*$")


def _yaml_top(path):
    """Cheap top-level name/description from a yaml file (no full parse needed)."""
    out = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:80]:
            m = _YAML_KEY.match(line)
            if m and m.group(1) not in out:
                out[m.group(1)] = m.group(2).strip("\"'")
    except OSError:
        pass
    return out


def _experiments():
    now = time.time()
    if _EXP_CACHE["data"] is not None and now - _EXP_CACHE["t"] < 10.0:
        return _EXP_CACHE["data"]
    exps = []
    root = ROOT / "experiments"
    paths = sorted(root.glob("presets/*.yaml")) + sorted(root.glob("*/experiment.yaml"))
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith("experiments/_lib/"):
            continue
        top = _yaml_top(p)
        exps.append({"path": rel, "name": top.get("name") or p.stem,
                     "description": top.get("description", "")})
    _EXP_CACHE.update(t=now, data=exps)
    return exps


def _presets():
    from rl.config import PRESETS
    return sorted(PRESETS.keys())


# ================================================================ routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/state")
def api_state():
    managed = MANAGER.snapshot()
    alive = {n: bool(m["alive"]) for n, m in managed.items()}
    return _ok(runs=scanner.scan_runs(alive), managed=managed,
               experiments=_experiments(), presets=_presets())


@app.get("/api/runs/<name>")
def api_run_detail(name):
    if not _check(name):
        return _err("bad run name", 400)
    d = RUNS / name
    if not d.is_dir():
        return _err(f"no such run: {name}", 404)
    resolved = scanner._read_json(d / "resolved_config.json") or {}
    cfg = resolved.get("config") or {}
    diff = {}
    if cfg:
        from rl.config import Config, config_to_dict
        defaults = json.loads(json.dumps(config_to_dict(Config())))   # tuples -> lists
        for k, v in cfg.items():
            if k not in defaults or defaults[k] != v:
                diff[k] = [defaults.get(k), v]
    runs = scanner.scan_runs(MANAGER.alive_map())
    summary = next((r for r in runs if r["name"] == name), None)
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    return _ok(name=name, summary=summary,
               source=scanner._read_json(d / "preset.json") or {},
               description=(summary or {}).get("source", {}).get("description", ""),
               resolved_config=cfg, n_envs=resolved.get("n_envs"),
               total_steps=resolved.get("total_steps"), config_diff=diff,
               probe=scanner._read_json(d / "gait_probe.json"),
               files=files, managed=MANAGER.snapshot().get(name))


@app.get("/api/runs/<name>/progress")
def api_run_progress(name):
    if not _check(name):
        return _err("bad run name", 400)
    try:
        since = max(0, int(request.args.get("since", 0)))
    except ValueError:
        return _err("since must be an integer", 400)
    return _ok(**scanner.read_progress(name, since))


@app.get("/api/runs/<name>/log")
def api_run_log(name):
    if not _check(name):
        return _err("bad run name", 400)
    path = RUNS / name / "train.log"
    try:
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return _err("offset must be an integer", 400)
    if not path.exists():
        return _ok(data="", offset=0, size=0)
    size = path.stat().st_size
    if offset > size:                     # log restarted/truncated — resync to the tail
        offset = max(0, size - LOG_CHUNK)
    with open(path, "rb") as f:
        f.seek(offset)
        blob = f.read(LOG_CHUNK)
    return _ok(data=blob.decode("utf-8", "replace"), offset=offset + len(blob), size=size)


@app.get("/api/runs/<name>/file/<fname>")
def api_run_file(name, fname):
    if not _check(name):
        return _err("bad run name", 400)
    if fname not in FILE_WHITELIST:
        return _err(f"file not allowed (whitelist: {sorted(FILE_WHITELIST)})", 403)
    d = RUNS / name
    if not (d / fname).exists():
        return _err(f"{fname} not present for {name}", 404)
    return send_from_directory(d, fname, conditional=True)   # conditional => video seeking


@app.post("/api/launch")
def api_launch():
    spec = request.get_json(force=True, silent=True) or {}
    try:
        info = MANAGER.launch(spec)
    except ValueError as e:
        return _err(e, 400)
    return _ok(**info)


@app.post("/api/runs/<name>/kill")
def api_kill(name):
    if not _check(name):
        return _err("bad run name", 400)
    try:
        info = MANAGER.kill(name)
    except ValueError as e:
        return _err(e, 400)
    return _ok(**info)


@app.post("/api/runs/<name>/tool")
def api_tool(name):
    if not _check(name):
        return _err("bad run name", 400)
    body = request.get_json(force=True, silent=True) or {}
    kind = body.get("kind")
    if kind not in TOOL_KINDS:
        return _err(f"kind must be one of {TOOL_KINDS}", 400)
    try:
        info = MANAGER.spawn_tool(name, kind)
    except ValueError as e:
        return _err(e, 400)
    return _ok(**info)
