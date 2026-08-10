"""Make the flat webui/ and fixed_gait/ modules importable, exactly as paths.py does at runtime."""
import os
import sys

WEBUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXED_GAIT = os.path.dirname(WEBUI)
REPO = os.path.dirname(FIXED_GAIT)
for p in (FIXED_GAIT, WEBUI, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes >10 s (the acceptance reproduction needs a "
                                       "real 10 s pre-trigger window)")
