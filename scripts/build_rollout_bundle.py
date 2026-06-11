"""Build the packaged GSM8K rollout bundle from an OLMES predictions file.

Provenance for data/gemma3_1b_it_gsm8k_test_k32.jsonl.gz:

    python scripts/build_rollout_bundle.py \
        --predictions /data/prnaidu/olmes_results/gemma3_1b_pass_rate_sampling/full_k32_20260421_gpus0_2_7/outputs/shard-00-gpu0/task-000-gsm8k-predictions.jsonl \
        --output data/gemma3_1b_it_gsm8k_test_k32.jsonl.gz

The predictions come from running `google/gemma-3-1b-it` (vLLM, temperature=0.6,
top_p=0.95, 32 rollouts per question, zero-shot chat format) on the full GSM8K
test set with the OLMES harness, task alias `gsm8k::olmo3:adapt`.

Per-rollout correctness is computed with the same `exact_match_hf_evaluate`
normalization OLMES uses for its doc-level `exact_match` metric (the predictions
file stores per-rollout extracted answers but only doc-level correctness, which
for plain ExactMatch covers only the first rollout). As a consistency check, the
recomputed correctness of rollout 0 is compared against the stored doc-level
`exact_match` for every question.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "olmes"))

from oe_eval.dependencies.hf_evaluate.exact_match import exact_match_hf_evaluate  # noqa: E402

# metric_kwargs of the gsm8k::olmo3:adapt task config (and ignore_case=True from
# GSM8K.make_metrics); used by OLMES ExactMatch for answer comparison.
REGEXES_TO_IGNORE = [",", "\\$", "(?s).*#### ", "\\.$"]
IGNORE_CASE = True


def is_correct(model_answer: str, label: str) -> bool:
    res = exact_match_hf_evaluate(
        predictions=[model_answer],
        references=[label],
        regexes_to_ignore=REGEXES_TO_IGNORE,
        ignore_case=IGNORE_CASE,
        ignore_punctuation=False,
    )
    return bool(res["exact_match"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from datasets import load_dataset

    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")

    mismatches = 0
    n_rows = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open() as fin, gzip.open(args.output, "wt", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            native_id = row["native_id"]
            doc = gsm8k_test[native_id]
            label = row["label"]
            short_answer = doc["answer"].split("####")[-1].strip()

            rollouts = []
            n_correct = 0
            for out in row["model_output"]:
                correct = is_correct(out.get("model_answer", ""), label)
                correct_flex = is_correct(out.get("model_answer_flex", ""), label)
                n_correct += int(correct)
                rollouts.append(
                    {
                        "continuation": out["continuation"],
                        "model_answer": out.get("model_answer", ""),
                        "model_answer_flex": out.get("model_answer_flex", ""),
                        "correct": correct,
                        "correct_flex": correct_flex,
                        "num_tokens": out.get("num_tokens"),
                    }
                )

            # Consistency check: OLMES doc-level exact_match uses rollout 0.
            stored = row["metrics"].get("exact_match")
            if stored is not None and bool(stored) != rollouts[0]["correct"]:
                mismatches += 1

            bundle_row = {
                "doc_id": row["doc_id"],
                "native_id": native_id,
                "question": doc["question"],
                "gold_answer": doc["answer"],
                "short_answer": short_answer,
                "label": label,
                "n_rollouts": len(rollouts),
                "n_correct": n_correct,
                "pass_rate": n_correct / len(rollouts) if rollouts else 0.0,
                "rollouts": rollouts,
            }
            fout.write(json.dumps(bundle_row, ensure_ascii=False) + "\n")
            n_rows += 1

    print(f"Wrote {n_rows} problems to {args.output}")
    print(f"Rollout-0 correctness mismatches vs stored exact_match: {mismatches}")
    if mismatches:
        raise SystemExit(f"{mismatches} mismatches — scoring does not replicate OLMES!")


if __name__ == "__main__":
    main()
