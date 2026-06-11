"""Per-rollout correctness scoring, identical to OLMES ExactMatch.

OLMES stores per-rollout extracted answers (``model_answer`` /
``model_answer_flex``) in the predictions file but only doc-level correctness.
This module recomputes per-rollout correctness with the exact same
``exact_match_hf_evaluate`` call and normalization parameters that the
``gsm8k::olmo3:adapt`` task config uses (verified to match the stored doc-level
``exact_match`` on all 1319 GSM8K test questions).
"""

from oe_eval.dependencies.hf_evaluate.exact_match import exact_match_hf_evaluate

# metric_kwargs from the gsm8k::olmo3:adapt task config, plus ignore_case=True
# which GSM8K.make_metrics hardcodes.
REGEXES_TO_IGNORE = [",", "\\$", "(?s).*#### ", "\\.$"]
IGNORE_CASE = True


def is_correct(model_answer: str, label: str) -> bool:
    res = exact_match_hf_evaluate(
        predictions=[model_answer or ""],
        references=[label],
        regexes_to_ignore=REGEXES_TO_IGNORE,
        ignore_case=IGNORE_CASE,
        ignore_punctuation=False,
    )
    return bool(res["exact_match"])


def score_prediction_row(row: dict, question: str, gold_answer: str) -> dict:
    """Convert one OLMES predictions.jsonl row into a bundle-format problem row."""
    label = row["label"]
    rollouts = []
    n_correct = 0
    for out in row["model_output"]:
        correct = is_correct(out.get("model_answer", ""), label)
        n_correct += int(correct)
        rollouts.append(
            {
                "continuation": out["continuation"],
                "model_answer": out.get("model_answer", ""),
                "model_answer_flex": out.get("model_answer_flex", ""),
                "correct": correct,
                "correct_flex": is_correct(out.get("model_answer_flex", ""), label),
                "num_tokens": out.get("num_tokens"),
            }
        )
    short_answer = gold_answer.split("####")[-1].strip() if "####" in gold_answer else gold_answer
    return {
        "doc_id": row["doc_id"],
        "native_id": row["native_id"],
        "question": question,
        "gold_answer": gold_answer,
        "short_answer": short_answer,
        "label": label,
        "n_rollouts": len(rollouts),
        "n_correct": n_correct,
        "pass_rate": n_correct / len(rollouts) if rollouts else 0.0,
        "rollouts": rollouts,
    }
