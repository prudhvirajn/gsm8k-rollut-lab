# GSM8K Rollout Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prudhvirajn/gsm8k-rollout-lab/blob/main/notebooks/gsm8k_rollout_lab_colab.ipynb)

Browse GSM8K problems with their packaged **Gemma 3 1B-IT rollouts**, **edit a problem**
(specifying a new gold answer), and **re-run rollouts through the exact
[OLMES](https://github.com/allenai/olmes) pipeline** that produced the originals — directly in
Google Colab.

## What's inside

- **`data/gemma3_1b_it_gsm8k_test_k32.jsonl.gz`** — all 1319 GSM8K test questions with 32
  rollouts each from `google/gemma-3-1b-it` (OLMES task `gsm8k::olmo3:adapt`: zero-shot chat
  format, boxed-latex CoT template, temperature 0.6, top_p 0.95, max_length 8192, vLLM).
  Each rollout carries the completion, the OLMES-extracted answer, and exact-match correctness;
  each problem carries its passrate (correct/32).
- **`olmes/`** — the vendored OLMES harness (Apache-2.0) the rollouts were generated with,
  pinned to the same versions (`vllm==0.11.0`, `transformers>=4.57`, `datasets 3.x`).
- **`gsm8k_rollout_lab/`** — the lab package:
  - `bundle.py` loads the packaged rollouts;
  - `viewer.py` has the ipywidgets problem browser and the edit-and-run panel;
  - `runner.py` keeps a vLLM model resident and runs edited problems through the unmodified
    OLMES code path (`load_task` → `build_all_requests` → chat-template conversion →
    `evaluate` → metrics), writing native OLMES output files per run;
  - `task_config.py` holds the canonical task/model config and writes edited problems as a
    local HuggingFace dataset that the stock OLMES GSM8K task loads via `dataset_path`;
  - `scoring.py` computes per-rollout correctness with the identical `exact_match_hf_evaluate`
    normalization OLMES uses (verified to match OLMES doc-level scores on all 1319 questions).
- **`notebooks/gsm8k_rollout_lab_colab.ipynb`** — the Colab notebook tying it together.
- **`scripts/build_rollout_bundle.py`** — provenance for the data bundle.

## Quickstart (Colab)

1. Click the badge above and select a GPU runtime (any Colab GPU works; the model is 1B params).
2. Add a Colab secret `HF_TOKEN` with a Hugging Face token that has access to
   [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) (accept the license on
   the model page).
3. Run the cells top to bottom. Setup takes ~5–10 min, model load ~2 min, then each
   32-rollout run takes ~1–2 min.

## Local usage

```python
pip install -e ./olmes[gpu]   # python 3.10+, CUDA GPU

import sys; sys.path.insert(0, "/path/to/gsm8k-rollout-lab")
from gsm8k_rollout_lab import load_bundle, GSM8KRolloutRunner

problems = load_bundle()
runner = GSM8KRolloutRunner()
result = runner.run(question="...", answer="42", n_rollouts=32)
print(result["pass_rate"])
```

## Replication notes

- New rollouts use the identical OLMES task config, prompt construction, chat template,
  sampling parameters, and answer scoring as the packaged ones.
- The original runs sampled at temperature 0.6 **without a seed**, so replication means
  *same distribution*, not bit-identical completions. Passing `seed=` to `runner.run`
  makes your own runs repeatable (OLMES per-repeat `generation_seeds`).
- Hardware differs from the original H100 runs; on pre-Ampere GPUs (T4/V100) the model runs
  in fp16 instead of bf16, which can shift passrates slightly.

## Licenses

- Lab code: MIT (`LICENSE`).
- `olmes/`: Apache-2.0, © Allen Institute for AI (`olmes/LICENSE`).
- GSM8K data: MIT, from [openai/grade-school-math](https://github.com/openai/grade-school-math)
  ([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)).
- The packaged completions were generated with [Gemma 3](https://ai.google.dev/gemma) and are
  subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
