"""Train the task-conditioned unified MTL policy.

This is a thin wrapper around ``scripts/mtl_train.py`` that keeps the original
equal/scheduled MTL entry points intact while defaulting to the conditioned env:

  - task ID is appended to policy observations
  - selected reward terms use task-specific weights
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _append_default_flag(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _append_default_flag("--task", "MTL-Conditioned-Unitree-Go2-AllTerrains-v0")
    _append_default_flag("--experiment_name", "unitree_go2_mtl_conditioned_seeds")
    runpy.run_path(str(Path(__file__).with_name("mtl_train.py")), run_name="__main__")
