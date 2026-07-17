"""Local web UI for RL training runs.

    .venv/Scripts/python.exe -m orchestrator.web [--port 8800]

Design (same discipline as fixed_gait/webui): plain Flask threaded on 127.0.0.1, a
RunManager background daemon owns the training/tool subprocesses, HTTP handlers only
read snapshots / post requests, every JSON reply is {ok: true, ...} | {ok: false, error}.
No auth — localhost only. No build step, no CDN: vanilla JS + canvas.
"""
