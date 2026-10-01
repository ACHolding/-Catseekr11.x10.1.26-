#!/usr/bin/env python3
"""Launcher — runs the fixed CatSeek R1 BitNet engine.

Your old logs came from this stub path. It now forwards to:
  #catseeekr110.1.26.py

That engine:
  - writes code to WORK_DIR on your device (file:// URL)
  - prints the full source in the terminal and CatSeek CODE log
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "#catseeekr110.1.26.py"

if not TARGET.exists():
    sys.stderr.write(f"Missing fixed engine: {TARGET}\n")
    sys.exit(1)

sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
