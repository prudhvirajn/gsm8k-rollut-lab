"""Canonical OLMES task/model configs for replicating the GSM8K rollouts.

``BASE_TASK_SPEC`` is the ``gsm8k::olmo3:adapt`` task spec with repeats=32 — the
exact config recorded in the packaged Gemma 3 1B-IT run's metrics.json
(zero-shot chat format, boxed-latex CoT template, temperature 0.6, top_p 0.95,
no pass_at_k metrics: pass rates are computed from per-rollout exact match).

``BASE_MODEL_CONFIG`` matches the model config recorded in the original run's
metrics.json.

``write_custom_dataset`` writes an edited problem as a local HuggingFace dataset
script so the unmodified OLMES GSM8K task can load it via ``dataset_path``
(the same mechanism used for counterfactual evals).
"""

import copy
import json
from pathlib import Path

BASE_TASK_SPEC = {
    "task_name": "gsm8k",
    "split": "test",
    "primary_metric": "exact_match_flex",
    "use_chat_format": True,
    "num_shots": 0,
    "chat_overrides": {
        "context_kwargs": {
            "description": None,
            "assistant_prefix": None,
            "fewshot_as_multiturn": False,
            "template": (
                "Solve the following grade school math word problem:\n{{question}}\n\n"
                'Show your work and conclude with "Therefore, the final answer is '
                '\\boxed{answer}." where answer is just the final number that solves '
                "the problem. E.g., if the answer is 6, conclude with \"Therefore, "
                'the final answer is \\boxed{6}."'
            ),
            "cot_style": "boxed_latex",
        },
        "generation_kwargs": {
            "stop_sequences": [],
            "max_gen_toks": 131072,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
            "truncate_context": False,
            "repeats": 32,
        },
        "metric_kwargs": {
            "answer_format_regex": "Therefore, the final answer is \\\\boxed\\{(.*)\\}",
            "answer_prefix_regexes": [
                "(?i)Therefore,? the final answer is",
                "(?i)Therefore,? the answer is",
                "(?i)the final answer is",
                "(?i)the answer is",
                "(?i)answer is",
                "(?i)answer is:",
                "(?i)answer:",
            ],
            "answer_regexes": [
                "[\\s\\S]*?\\\\boxed\\{(.*?)\\}[\\s\\S]*?",
                "([-+]?\\d*\\.\\d+|\\d+)",
                "(.*)\\.?",
            ],
            "answer_format_correct_cutoff": 0.4,
        },
    },
    "metadata": {"alias": "gsm8k::olmo3:adapt"},
}

BASE_MODEL_CONFIG = {
    "model": "google/gemma-3-1b-it",
    "revision": None,
    "trust_remote_code": True,
    "max_length": 8192,
    "model_path": None,
    "model_type": "vllm",
    "gpu_memory_utilization": 0.7,
    "max_num_seqs": 32,
    "tensor_parallel_size": 1,
    "add_bos_token": True,
    "chat_model": True,
}

DATASET_SCRIPT = '''\
import json
from pathlib import Path

import datasets


class Gsm8kCustom(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "short_answer": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        data_path = Path("__DATA_PATH__")
        return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"path": data_path})]

    def _generate_examples(self, path):
        with open(path) as f:
            for idx, line in enumerate(f):
                if line.strip():
                    yield idx, json.loads(line)
'''


def normalize_gold_answer(answer: str) -> str:
    """Accept either a bare final answer ("42") or full GSM8K-style CoT with '#### 42'."""
    answer = str(answer).strip()
    if "####" not in answer:
        answer = f"#### {answer}"
    return answer


def write_custom_dataset(dataset_dir: Path, problems: list) -> Path:
    """Write edited problems as a local HF dataset script; returns the script path.

    Each problem is a dict with at least ``question`` and ``answer`` (the answer
    may be a bare final number or full CoT ending in '#### <number>').
    """
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, problem in enumerate(problems):
        answer = normalize_gold_answer(problem["answer"])
        rows.append(
            {
                "id": str(problem.get("id", i)),
                "question": str(problem["question"]).strip(),
                "answer": answer,
                "short_answer": answer.split("####")[-1].strip(),
            }
        )
    data_path = dataset_dir / "test.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    script_path = dataset_dir / "gsm8k_custom.py"
    script_path.write_text(DATASET_SCRIPT.replace("__DATA_PATH__", str(data_path)))
    return script_path


def make_task_spec(
    dataset_script: Path,
    n_rollouts: int = 32,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_gen_toks: int = 131072,
    seed: int = None,
) -> dict:
    """Build the task spec for a custom run: the canonical k32 spec with the
    dataset pointed at the local script and generation settings overridden."""
    spec = copy.deepcopy(BASE_TASK_SPEC)
    spec["dataset_path"] = str(dataset_script)
    spec["dataset_name"] = None
    spec["native_id_field"] = "id"
    gen = spec["chat_overrides"]["generation_kwargs"]
    gen["repeats"] = int(n_rollouts)
    gen["max_gen_toks"] = int(max_gen_toks)
    if temperature == 0:
        gen["temperature"] = 0.0
        gen["do_sample"] = False
        gen.pop("top_p", None)
    else:
        gen["temperature"] = float(temperature)
        gen["top_p"] = float(top_p)
    if seed is not None:
        # OLMES-native per-repeat seeding; note requests are then batched per
        # seed, which is slower than one unseeded batch.
        gen["generation_seeds"] = [int(seed) + i for i in range(int(n_rollouts))]
    return spec
