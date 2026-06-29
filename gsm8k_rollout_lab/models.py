"""Model presets and GPU-aware vLLM config for the rollout runner.

``MODEL_PRESETS`` are convenience entries for the notebook's model picker; any
Hugging Face model id works too (pass it straight to :func:`build_model_overrides`).
All presets use **ungated** unsloth mirrors so no Hugging Face token is needed.

``build_model_overrides`` turns a preset name *or* a raw HF id into a
``model_overrides`` dict for :class:`GSM8KRolloutRunner`, picking dtype and engine
settings from the detected GPU:
  - compute capability >= 8 (A100/L4/H100): bf16, FlashAttention (vLLM default),
    higher memory utilization.
  - older GPUs (T4/V100, capability < 8): float32 (these have no bf16, and vLLM's
    Gemma 3 forbids fp16), ``enforce_eager``, and the Triton attention backend.
    vLLM would otherwise pick FlexAttention for Gemma 3 on these GPUs, which
    crashes with a CUDA-graph block-mask error; Triton runs on Turing/Volta.
"""

import os

# Each preset merges onto BASE_MODEL_CONFIG. ``min_capability`` is the lowest GPU
# compute-capability major version that fits the model in the chosen dtype on a
# single ~40GB GPU; ``note`` is shown by the notebook picker. ``thinking`` marks
# hybrid reasoning models (Qwen3) that emit <think>...</think> by default.
MODEL_PRESETS = {
    "gemma-3-1b-it": {
        "model": "unsloth/gemma-3-1b-it",
        "max_length": 8192,
        "add_bos_token": True,
        "note": "Default. Matches the packaged rollouts. Fits any GPU.",
    },
    "gemma-3-4b-it": {
        "model": "unsloth/gemma-3-4b-it",
        "max_length": 8192,
        "add_bos_token": True,
        "note": "Fits a T4 (fp32) tightly; comfortable on L4/A100.",
    },
    "gemma-3-12b-it": {
        "model": "unsloth/gemma-3-12b-it",
        "max_length": 8192,
        "add_bos_token": True,
        "max_num_seqs": 16,
        "note": "Needs an L4/A100 (bf16). Won't fit a T4.",
    },
    "Qwen3-1.7B": {
        "model": "unsloth/Qwen3-1.7B",
        "max_length": 16384,
        "add_bos_token": False,
        "thinking": True,
        "note": "Hybrid thinking. Fits any GPU.",
    },
    "Qwen3-4B": {
        "model": "unsloth/Qwen3-4B",
        "max_length": 16384,
        "add_bos_token": False,
        "thinking": True,
        "note": "Hybrid thinking. Fits a T4 (fp32) tightly; comfortable on L4/A100.",
    },
    "Qwen3-8B": {
        "model": "unsloth/Qwen3-8B",
        "max_length": 16384,
        "add_bos_token": False,
        "thinking": True,
        "max_num_seqs": 16,
        "note": "Hybrid thinking. Needs an L4/A100 (bf16).",
    },
    "Qwen3-14B": {
        "model": "unsloth/Qwen3-14B",
        "max_length": 16384,
        "add_bos_token": False,
        "thinking": True,
        "max_num_seqs": 8,
        "note": "Hybrid thinking. Largest that fits one 40GB A100 in bf16 (~28GB).",
    },
}

# Keys that describe a preset but are not valid vLLM/model-config overrides.
_META_KEYS = ("note", "thinking", "min_capability")


def _gpu_capability_major() -> int:
    """Major compute-capability of the current CUDA device (0 if no GPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()[0]
    except Exception:
        pass
    return 0


def silence_logs() -> None:
    """Quiet noisy third-party INFO/WARNING logging for a clean notebook UI.

    These messages (OLMES `INFO:root`, HF datasets fingerprint warnings, etc.)
    go through Python ``logging`` with handlers bound to the real stderr, so the
    panel's stdout/stderr capture can't suppress them — we raise the loggers'
    levels instead. Call after OLMES is imported (it configures root logging).
    """
    import logging
    import warnings

    logging.getLogger().setLevel(logging.WARNING)  # drop INFO:root from OLMES
    for name in ("datasets", "transformers", "vllm", "filelock", "fsspec", "httpx"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")


def configure_engine_env(capability_major: int = None) -> None:
    """Set vLLM env vars that must precede engine init, based on the GPU.

    On pre-Ampere GPUs (capability < 8) force the Triton attention backend so
    vLLM doesn't fall back to FlexAttention (which crashes with Gemma 3). A
    user-set ``VLLM_ATTENTION_BACKEND`` is respected. Safe to call repeatedly;
    must run before the vLLM engine is constructed (i.e. before load_model).
    """
    cap = _gpu_capability_major() if capability_major is None else capability_major
    if 0 < cap < 8:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")


def is_thinking_model(model: str) -> bool:
    """True if ``model`` (preset name or HF id) is a known hybrid-thinking model."""
    if model in MODEL_PRESETS:
        return bool(MODEL_PRESETS[model].get("thinking"))
    return "qwen3" in model.lower()


def build_model_overrides(model: str, dtype: str = None, **extra) -> dict:
    """Build a ``model_overrides`` dict for a preset name or raw HF model id.

    ``dtype`` forces a dtype (else it's chosen from the GPU). Extra keyword args
    override anything (e.g. ``gpu_memory_utilization=0.95``, ``max_num_seqs=4``).
    """
    if model in MODEL_PRESETS:
        ov = {k: v for k, v in MODEL_PRESETS[model].items() if k not in _META_KEYS}
    else:
        ov = {"model": model}

    cap = _gpu_capability_major()
    if dtype is not None:
        ov["dtype"] = dtype
    elif cap >= 8:
        ov.setdefault("dtype", "bfloat16")  # A100/L4/H100
    else:
        # T4/V100: no bf16, and vLLM's Gemma 3 rejects fp16 -> fp32 is the only option.
        ov["dtype"] = "float32"
        ov.setdefault("enforce_eager", True)  # avoid CUDA-graph crashes on older GPUs
        # fp32 doubles KV-cache/activation memory; on a 16GB T4 the default 32
        # concurrent sequences can OOM, so cap concurrency unless overridden.
        ov.setdefault("max_num_seqs", 8)
    ov.setdefault("gpu_memory_utilization", 0.9 if cap >= 8 else 0.85)
    # Steer the attention backend away from FlexAttention on old GPUs (must be
    # set before the engine is built, which happens at runner creation).
    configure_engine_env(cap)
    ov.update(extra)
    return ov
