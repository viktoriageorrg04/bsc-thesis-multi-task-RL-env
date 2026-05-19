"""Compatibility wrapper for the cross-evaluation matrix exporter.

Prefer:
  python analysis/export_eval_matrix.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    raise SystemExit(runpy.run_path(str(Path(__file__).parents[1] / "analysis" / "export_eval_matrix.py"), run_name="__main__"))
