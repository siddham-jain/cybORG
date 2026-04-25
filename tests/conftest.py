"""Shared pytest configuration.

Adds the ``cyborg_env`` package to ``sys.path`` so tests can import the
environment without first installing the project. Tests run against the
in-process environment class — no HTTP server, no Docker.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "cyborg_env"

# Add the project root so ``cyborg_env.*`` imports resolve.
sys.path.insert(0, str(ROOT))
# Also add the package itself so ``server.*`` and ``models`` style imports
# match the in-container layout.
sys.path.insert(0, str(ENV_ROOT))
