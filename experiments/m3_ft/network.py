"""Policy/value network for m3_ft — the architecture escape hatch.

Most experiments never need this file: set `policy.hidden` / `policy.activation` in
experiment.yaml and the framework builds a standard MLP. It's included here only to
show the override point. This one just reproduces the [256, 256]-tanh default
explicitly (shared actor/critic trunk sizes; separate state-independent log_std,
clamped at cfg.max_log_std by the entropy schedule).

Return an SB3-compatible `policy_kwargs`, OR, for a genuinely custom net (e.g. an LSTM
over the history, or a separate CPG head), define and return a policy class instead.
"""
import torch.nn as nn


def policy_kwargs(cfg) -> dict:
    return dict(
        net_arch=list(cfg.policy_hidden),        # [256, 256]
        activation_fn=nn.Tanh,
        log_std_init=0.0,                        # std = 1 at start; annealed via entropy schedule
    )
