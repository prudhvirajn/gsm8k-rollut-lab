"""In-process OLMES rollout runner.

Mirrors the per-task body of ``oe_eval.run_eval.run_eval`` (task loading,
request construction, chat-template conversion, ``evaluate``, metric
computation, and predictions/metrics file output all reuse the OLMES functions
directly), but keeps the vLLM model loaded between runs so editing a problem
and re-rolling does not pay the model load each time.
"""

import copy
import datetime
import json
import re
import time
from pathlib import Path

from oe_eval.default_configs import MODEL_DEFAULTS, TASK_DEFAULTS
from oe_eval.run_eval import (
    compute_save_metrics,
    convert_chat_instance,
    evaluate,
    load_task,
)
from oe_eval.utilities.model_utils import load_model
from oe_eval.utils import hash_dict

from .scoring import score_prediction_row
from .task_config import (
    BASE_MODEL_CONFIG,
    make_task_spec,
    normalize_gold_answer,
    write_custom_dataset,
)


class GSM8KRolloutRunner:
    """Loads the model once; ``run()`` evaluates one edited problem at a time."""

    def __init__(self, model_overrides: dict = None, work_dir: str = "rollout_runs"):
        self.model_config = copy.deepcopy(BASE_MODEL_CONFIG)
        if model_overrides:
            self.model_config.update(model_overrides)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        model_load_config = dict(self.model_config)
        model_load_config["batch_size"] = "auto"
        self.model = load_model(model_load_config)

    def run(
        self,
        question: str,
        answer: str,
        n_rollouts: int = 32,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_gen_toks: int = 131072,
        seed: int = None,
        run_name: str = None,
        system_prompt: str = None,
        thinking: bool = None,
    ) -> dict:
        """Run rollouts for one (possibly edited) GSM8K problem.

        ``answer`` may be a bare final number ("42") or a full GSM8K-style CoT
        ending in "#### 42". Returns a bundle-format problem dict (question,
        pass_rate, per-rollout continuations/answers/correctness) with the
        OLMES task metrics attached under ``"olmes_metrics"`` and the run
        directory under ``"run_dir"``.

        ``system_prompt`` is prepended as a chat system message (the original
        runs used none). ``thinking`` controls hybrid reasoning models (e.g.
        Qwen3) via their ``/think`` `/` ``/no_think`` soft switch: ``True`` forces
        thinking on, ``False`` off, ``None`` leaves the model default. Thinking
        models reason inside ``<think>...</think>`` before the answer, so give
        them a generous ``max_gen_toks``.
        """
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", run_name) if run_name else "run"
        run_dir = self.work_dir / f"{stamp}_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)

        dataset_script = write_custom_dataset(
            run_dir / "dataset", [{"id": slug, "question": question, "answer": answer}]
        )
        task_spec = make_task_spec(
            dataset_script,
            n_rollouts=n_rollouts,
            temperature=temperature,
            top_p=top_p,
            max_gen_toks=max_gen_toks,
            seed=seed,
        )
        # Apply the system prompt / thinking switch onto the chat context. Done
        # here (not in make_task_spec) so the canonical task-spec builder stays
        # untouched. Qwen3 honors /think and /no_think in the system message.
        sys_prompt = (system_prompt or "").strip()
        if thinking is True:
            sys_prompt = f"{sys_prompt} /think".strip()
        elif thinking is False:
            sys_prompt = f"{sys_prompt} /no_think".strip()
        if sys_prompt:
            task_spec["chat_overrides"]["context_kwargs"]["system_prompt"] = sys_prompt
        (run_dir / "task_spec.json").write_text(json.dumps(task_spec, indent=2))

        # --- Below mirrors oe_eval.run_eval.run_eval's per-task processing ---
        start_time = time.time()
        task = load_task(copy.deepcopy(task_spec), output_dir=str(run_dir))
        task.download()
        task.set_model(self.model, self.model_config)
        task.build_all_requests()
        task_instances = task._instances or []
        if not task_instances:
            raise RuntimeError("OLMES task constructed no requests")
        eval_requests_raw = [ins.to_dict() for ins in task_instances]

        if task.task_config.get("use_chat_format"):
            for ins in task_instances:
                convert_chat_instance(self.model, ins, self.model_config.get("chat_template"))

        results_for_requests = evaluate(
            model=self.model,
            instances=task_instances,
            task_config=task.task_config,
            model_config=self.model_config,
        )

        task.model = None
        task.make_metrics()

        full_config = {
            "task_name": task.task_name,
            "task_hash": hash_dict(task.task_config, TASK_DEFAULTS)["hash"],
            "model_hash": hash_dict(self.model_config, MODEL_DEFAULTS)["hash"],
            "model_config": self.model_config,
            "task_config": task.task_config,
            "compute_config": {"batch_size": "auto"},
            "processing_time": time.time() - start_time,
            "current_date": datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "num_instances": 0,
            "beaker_info": {},
        }
        compute_config = {
            "output_dir": str(run_dir),
            "num_recorded_inputs": 1,
            "gsheet": None,
        }
        metrics = compute_save_metrics(
            0, task, full_config, compute_config, eval_requests_raw, results_for_requests
        )
        # --- End of mirrored run_eval processing ---

        predictions_path = next(run_dir.glob("task-000-*-predictions.jsonl"))
        prediction_row = json.loads(predictions_path.read_text().splitlines()[0])
        result = score_prediction_row(
            prediction_row, question=question, gold_answer=normalize_gold_answer(answer)
        )
        result["olmes_metrics"] = metrics.get("metrics", {})
        result["run_dir"] = str(run_dir)
        result["generation_settings"] = {
            "model": self.model_config["model"],
            "n_rollouts": n_rollouts,
            "temperature": temperature,
            "top_p": top_p,
            "max_gen_toks": max_gen_toks,
            "seed": seed,
        }
        (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
