from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Some locked-down Windows desktops deny access to the default user TEMP
# directory. Keep pytest's tmp_path root inside the workspace so local and
# Codex runs behave like CI without requiring machine-level permissions.
PYTEST_TMP = ROOT / ".pytest-tmp"
PYTEST_TMP.mkdir(exist_ok=True)
os.environ.setdefault("TMP", str(PYTEST_TMP))
os.environ.setdefault("TEMP", str(PYTEST_TMP))
os.environ.setdefault("TMPDIR", str(PYTEST_TMP))
tempfile.tempdir = str(PYTEST_TMP)


@pytest.fixture
def tmp_path(request) -> Path:
    """Workspace-local replacement for pytest's tmp_path on locked Windows hosts."""

    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request.node.name)[:80]
    path = ROOT / ".pytest-manual-tmp" / f"{safe_name}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path
