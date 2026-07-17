"""framework — the runtime behind the experiments/ schema.

schema.py   what an experiment IS (typed pydantic models, authoring shorthands)
loader.py   YAML folder/file -> validated Experiment (inherits deep-merge)
compile.py  Experiment -> rl.config.Config (capability-gated) + run metadata
validate.py CLI preflight: `python -m framework.validate experiments/<name>`
"""
