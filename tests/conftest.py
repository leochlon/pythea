import os
import sys
from pathlib import Path

import pytest

# Allow `pytest` to run without requiring an editable install.
# (In CI you can still `pip install -e .[dev]` and this will be harmless.)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end tests (requires THEA_BASE_URL)")
