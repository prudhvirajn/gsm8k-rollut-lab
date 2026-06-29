"""GSM8K Rollout Lab: browse OLMES GSM8K rollouts, edit problems, re-roll."""

# Ensure the vendored OLMES (`<repo>/olmes/oe_eval`) is importable even when the
# `pip install -e ./olmes` editable registration didn't take effect — a recurring
# flake on Colab where the editable .pth isn't picked up, surfacing as
# `ModuleNotFoundError: No module named 'oe_eval'`. Resolving it from the repo
# layout here makes the import deterministic (the `-e` install is still what
# provides OLMES's dependencies like vLLM).
import sys as _sys
from pathlib import Path as _Path

_olmes_dir = _Path(__file__).resolve().parent.parent / "olmes"
if _olmes_dir.is_dir() and str(_olmes_dir) not in _sys.path:
    _sys.path.insert(0, str(_olmes_dir))

from .bundle import DEFAULT_BUNDLE, load_bundle
from .models import MODEL_PRESETS, build_model_overrides, is_thinking_model
from .task_config import BASE_MODEL_CONFIG, BASE_TASK_SPEC

__all__ = [
    "DEFAULT_BUNDLE",
    "load_bundle",
    "BASE_MODEL_CONFIG",
    "BASE_TASK_SPEC",
    "MODEL_PRESETS",
    "build_model_overrides",
    "is_thinking_model",
    "GSM8KRolloutRunner",
    "ProblemBrowser",
    "EditRunPanel",
]


def __getattr__(name):
    # Lazy imports: runner needs oe_eval (heavy), viewer needs ipywidgets.
    if name == "GSM8KRolloutRunner":
        from .runner import GSM8KRolloutRunner

        return GSM8KRolloutRunner
    if name in ("ProblemBrowser", "EditRunPanel"):
        from . import viewer

        return getattr(viewer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
