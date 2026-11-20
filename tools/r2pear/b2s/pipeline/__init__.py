"""Composable orchestration for the Blender-to-Sim (b2s) pipeline.

The modules in :mod:`tools.r2pear.b2s` keep the domain algorithms. This
package owns only the lifecycle and stage boundaries so new inputs or output
stages can be added without turning the command-line entry point into another
large dispatcher.
"""

import sys
from pathlib import Path

# The legacy domain modules intentionally use short imports (``common``,
# ``demo`` …) because they are also executable as standalone scripts. Add the
# b2s directory when this package is imported through ``tools.r2pear`` so the
# new Python API works from either entry point.
_B2S_DIR = Path(__file__).resolve().parent.parent
if str(_B2S_DIR) not in sys.path:
    sys.path.insert(0, str(_B2S_DIR))

from .models import PipelineContext, PipelineOptions
from .runner import run_pipeline

__all__ = ["PipelineContext", "PipelineOptions", "run_pipeline"]
