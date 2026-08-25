import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_SCRUB = (
    "R2_S3_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "ONIQ_ALLOW_CPU_FALLBACK",
    "RUNPOD_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ambient credentials or fallback flags must never flip a test."""
    for name in _SCRUB:
        monkeypatch.delenv(name, raising=False)
