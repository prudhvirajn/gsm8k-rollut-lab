"""GSM8K Rollout Lab: browse OLMES GSM8K rollouts, edit problems, re-roll."""

from .bundle import DEFAULT_BUNDLE, load_bundle
from .task_config import BASE_MODEL_CONFIG, BASE_TASK_SPEC

__all__ = [
    "DEFAULT_BUNDLE",
    "load_bundle",
    "BASE_MODEL_CONFIG",
    "BASE_TASK_SPEC",
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
