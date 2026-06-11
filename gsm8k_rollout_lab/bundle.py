"""Load the packaged GSM8K rollout bundle."""

import gzip
import json
from pathlib import Path

DEFAULT_BUNDLE = Path(__file__).resolve().parent.parent / "data" / "gemma3_1b_it_gsm8k_test_k32.jsonl.gz"


def load_bundle(path=DEFAULT_BUNDLE) -> list:
    """Returns a list of problem dicts (question, gold answer, pass_rate, 32 rollouts)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
