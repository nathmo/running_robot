"""RunManager — the one background daemon that owns training/tool subprocesses.

HTTP handlers never touch Popen objects directly: they call launch()/kill()/spawn_tool()
(which validate + mutate under a lock) and read snapshot(). A reaper thread notices
process exits, closes log file handles and post-processes gait_probe stdout into
rl/runs/NAME/gait_probe.json.
"""
from __future__ import annotations

import atexit
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
RUNS = ROOT / "rl" / "runs"
PY = sys.executable                                  # the .venv python running this server
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TOOL_KINDS = ("evaluate", "gait_probe")

if str(ROOT) not in sys.path:                        # rl/ + framework/ imports below
    sys.path.insert(0, str(ROOT))


def check_name(name):
    """Return the name if it is a safe run-folder name, else raise ValueError."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError("run name must match [A-Za-z0-9_.-]+")
    return name


class RunManager:
    """Owns every subprocess this server started. All state behind one lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.procs = {}      # name -> {popen, cmd, started, log_path, log_fh, killed, rc, ended}
        self.tools = {}      # name -> {kind, popen, cmd, started, out_fh, rc, ended, error}
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True,
                                        name="runmanager-reaper")
        self._reaper.start()
        atexit.register(self._close_all)

    # ------------------------------------------------------------------ launch
    def launch(self, spec):
        """Validate a launch spec, build the rl.train command and start it.

        spec: {base: "preset:NAME"|"experiment:PATH", name, description (required),
               steps?, n_envs?, subproc?, overrides?: {field: value}}
        Raises ValueError with a user-facing message on any bad input (-> HTTP 400).
        """
        spec = spec or {}
        name = check_name(spec.get("name") or "")
        run_dir = RUNS / name
        if run_dir.exists():
            raise ValueError(f"rl/runs/{name} already exists — pick a new name")
        description = str(spec.get("description") or "").strip()
        if not description:
            raise ValueError("description is required (say what this run is trying)")

        base = str(spec.get("base") or "")
        kind, _, target = base.partition(":")
        if kind not in ("preset", "experiment") or not target:
            raise ValueError('base must be "preset:NAME" or "experiment:PATH"')
        if kind == "experiment":
            exp_path = (ROOT / target)
            if not exp_path.exists():
                raise ValueError(f"experiment path not found: {target}")

        overrides = spec.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be an object {field: value}")

        base_flags = None
        launch_cfg_path = None
        if overrides:
            # resolve base -> Config in-process so a typo'd field fails NOW, not mid-train
            from rl.config import apply_overrides, config_to_dict, get_config
            if kind == "preset":
                try:
                    cfg = get_config(target)
                except KeyError:
                    raise ValueError(f"unknown preset '{target}'")
            else:
                from framework.compile import compile_experiment
                from framework.loader import load_experiment
                try:
                    cfg = compile_experiment(load_experiment(str(ROOT / target))).config
                except Exception as e:
                    raise ValueError(f"experiment failed to compile: {e}")
            try:
                cfg2 = apply_overrides(cfg, overrides)
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(str(e))
            run_dir.mkdir(parents=True, exist_ok=True)
            launch_cfg_path = run_dir / "launch_config.json"
            launch_cfg_path.write_text(json.dumps({"config": config_to_dict(cfg2)}, indent=1))
            base_flags = ["--config", str(launch_cfg_path.relative_to(ROOT))]
        elif kind == "preset":
            from rl.config import PRESETS
            if target not in PRESETS:
                raise ValueError(f"unknown preset '{target}'")
            base_flags = ["--preset", target]
        else:
            base_flags = ["--experiment", target]

        cmd = [PY, "-m", "rl.train", *base_flags, "--name", name, "--no-progress"]
        if spec.get("steps"):
            cmd += ["--steps", str(int(spec["steps"]))]
        if spec.get("n_envs"):
            cmd += ["--n-envs", str(int(spec["n_envs"]))]
        if spec.get("subproc"):
            cmd += ["--subproc"]

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "description.txt").write_text(description, encoding="utf-8")

        with self.lock:
            live = self.procs.get(name)
            if live and live["popen"].poll() is None:
                raise ValueError(f"a managed process for '{name}' is already running")
            log_path = run_dir / "train.log"
            fh = open(log_path, "ab")
            fh.write((f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                      f"$ {' '.join(cmd)}\n").encode())
            fh.flush()
            try:
                popen = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
            except OSError as e:
                fh.close()
                raise ValueError(f"failed to start rl.train: {e}")
            self.procs[name] = {"popen": popen, "cmd": cmd, "started": time.time(),
                                "log_path": str(log_path), "log_fh": fh,
                                "killed": False, "rc": None, "ended": None}
        return {"name": name, "pid": popen.pid, "cmd": cmd,
                "launch_config": str(launch_cfg_path) if launch_cfg_path else None}

    # ------------------------------------------------------------------ kill
    def kill(self, name):
        """taskkill /T /F the whole tree (SubprocVecEnv workers included)."""
        check_name(name)
        with self.lock:
            ent = self.procs.get(name)
            if ent is None:
                raise ValueError(f"'{name}' is not managed by this server (no pid to kill)")
            popen = ent["popen"]
            if popen.poll() is not None:
                ent["killed"] = ent["killed"] or False
                return {"already_dead": True, "rc": popen.returncode}
            ent["killed"] = True
            pid = popen.pid
        if sys.platform == "win32":
            out = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                 capture_output=True, text=True)
            detail = (out.stdout or "") + (out.stderr or "")
        else:
            popen.terminate()
            detail = "terminated"
        try:
            popen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._finalize(name)
        return {"killed": True, "pid": pid, "detail": detail.strip()}

    # ------------------------------------------------------------------ tools
    def spawn_tool(self, name, kind):
        """Run evaluate (writes eval.mp4) or gait_probe (stdout -> gait_probe.json).
        One tool at a time per run."""
        check_name(name)
        if kind not in TOOL_KINDS:
            raise ValueError(f"kind must be one of {TOOL_KINDS}")
        run_dir = RUNS / name
        if not run_dir.is_dir():
            raise ValueError(f"rl/runs/{name} does not exist")
        rel = f"rl/runs/{name}"
        if kind == "evaluate":
            cmd = [PY, "-m", "rl.evaluate", "--run", rel, "--episodes", "3",
                   "--video", f"{rel}/eval.mp4"]
        else:
            cmd = [PY, "-m", "rl.gait_probe", "--run", rel]
        with self.lock:
            live = self.tools.get(name)
            if live and live["popen"].poll() is None:
                raise ValueError(f"a {live['kind']} is already running for '{name}'")
            out_path = run_dir / f"{kind}.log"
            fh = open(out_path, "wb")     # stdout only — gait_probe JSON must stay clean
            err_fh = open(run_dir / f"{kind}.stderr.log", "wb")
            try:
                popen = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=err_fh,
                                         stdin=subprocess.DEVNULL)
            except OSError as e:
                fh.close()
                err_fh.close()
                raise ValueError(f"failed to start {kind}: {e}")
            self.tools[name] = {"kind": kind, "popen": popen, "cmd": cmd,
                                "started": time.time(), "out_fh": fh, "err_fh": err_fh,
                                "out_path": str(out_path), "rc": None, "ended": None,
                                "error": None}
        return {"name": name, "kind": kind, "pid": popen.pid}

    # ------------------------------------------------------------------ snapshots
    def snapshot(self):
        """Read-only view for /api/state: managed process + tool info per run."""
        with self.lock:
            out = {}
            for name, e in self.procs.items():
                out[name] = {"pid": e["popen"].pid, "alive": e["popen"].poll() is None,
                             "rc": e["popen"].returncode, "killed": e["killed"],
                             "started": e["started"], "ended": e["ended"],
                             "cmd": " ".join(e["cmd"]), "tool": None}
            for name, t in self.tools.items():
                info = {"kind": t["kind"], "alive": t["popen"].poll() is None,
                        "rc": t["popen"].returncode, "started": t["started"],
                        "ended": t["ended"], "error": t["error"]}
                out.setdefault(name, {"pid": None, "alive": False, "rc": None, "killed": False,
                                      "started": None, "ended": None, "cmd": None,
                                      "tool": None})["tool"] = info
            return out

    def alive_map(self):
        with self.lock:
            return {n: e["popen"].poll() is None for n, e in self.procs.items()}

    # ------------------------------------------------------------------ reaper
    def _reap_loop(self):
        while True:
            time.sleep(1.5)
            with self.lock:
                train_done = [n for n, e in self.procs.items()
                              if e["popen"].poll() is not None and e["log_fh"] is not None]
                tool_done = [n for n, t in self.tools.items()
                             if t["popen"].poll() is not None and t["out_fh"] is not None]
            for n in train_done:
                self._finalize(n)
            for n in tool_done:
                self._finalize_tool(n)

    def _finalize(self, name):
        with self.lock:
            e = self.procs.get(name)
            if e is None or e["popen"].poll() is None:
                return
            if e["log_fh"] is not None:
                try:
                    e["log_fh"].close()
                except OSError:
                    pass
                e["log_fh"] = None
            e["rc"] = e["popen"].returncode
            e["ended"] = e["ended"] or time.time()

    def _finalize_tool(self, name):
        with self.lock:
            t = self.tools.get(name)
            if t is None or t["popen"].poll() is None or t["out_fh"] is None:
                return
            for k in ("out_fh", "err_fh"):
                try:
                    t[k].close()
                except OSError:
                    pass
                t[k] = None
            t["rc"] = t["popen"].returncode
            t["ended"] = time.time()
            kind, out_path = t["kind"], t["out_path"]
        if kind == "gait_probe":
            err = self._write_probe_json(name, out_path)
            if err:
                with self.lock:
                    self.tools[name]["error"] = err

    @staticmethod
    def _write_probe_json(name, out_path):
        """Strip any non-JSON leading lines before the first '{' and save gait_probe.json."""
        try:
            text = Path(out_path).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"cannot read probe output: {e}"
        i = text.find("{")
        if i < 0:
            return "gait_probe printed no JSON object"
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[i:])
        except ValueError as e:
            return f"gait_probe output is not valid JSON: {e}"
        (RUNS / name / "gait_probe.json").write_text(json.dumps(obj, indent=2))
        return None

    def _close_all(self):
        with self.lock:
            for e in self.procs.values():
                if e["log_fh"] is not None:
                    try:
                        e["log_fh"].close()
                    except OSError:
                        pass
            for t in self.tools.values():
                for k in ("out_fh", "err_fh"):
                    if t.get(k) is not None:
                        try:
                            t[k].close()
                        except OSError:
                            pass
