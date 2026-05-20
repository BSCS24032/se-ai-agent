"""Pytest configuration - ensure the project root is on sys.path."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make sure tests never accidentally hit a real provider by default.
os.environ.setdefault("LLM_PROVIDER", "none")
