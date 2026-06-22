"""ipywidgets UI: browse problems/rollouts, edit a problem, run new rollouts."""

import html
import json
import traceback
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

_BADGE = (
    "<span style='display:inline-block;padding:2px 8px;border-radius:10px;"
    "color:white;font-weight:bold;background:{bg}'>{text}</span>"
)


def _passrate_badge(pass_rate: float, n_correct: int, n_rollouts: int) -> str:
    bg = "#2e7d32" if pass_rate >= 0.75 else "#f9a825" if pass_rate >= 0.25 else "#c62828"
    return _BADGE.format(bg=bg, text=f"passrate {n_correct}/{n_rollouts} = {pass_rate:.2f}")


def _problem_html(problem: dict) -> str:
    badge = _passrate_badge(problem["pass_rate"], problem["n_correct"], problem["n_rollouts"])
    return (
        f"<div style='line-height:1.5'>{badge}"
        f"&nbsp;&nbsp;<b>gold answer: {html.escape(str(problem['short_answer']))}</b>"
        f"<div style='margin-top:8px;padding:10px;background:#f5f5f5;border-radius:6px;"
        f"white-space:pre-wrap;color:#111'>{html.escape(problem['question'])}</div></div>"
    )


def _rollout_html(rollout: dict, index: int) -> str:
    ok = rollout["correct"]
    badge = _BADGE.format(
        bg="#2e7d32" if ok else "#c62828",
        text=("✓ correct" if ok else "✗ wrong") + f" — extracted: {html.escape(str(rollout['model_answer']) or '∅')}",
    )
    tokens = rollout.get("num_tokens")
    token_info = f"&nbsp;&nbsp;<i>{tokens} tokens</i>" if tokens is not None else ""
    return (
        f"<div style='line-height:1.5'><b>rollout {index}</b>&nbsp;&nbsp;{badge}{token_info}"
        f"<div style='margin-top:6px;padding:10px;background:#fbfbfb;border:1px solid #e0e0e0;"
        f"border-radius:6px;white-space:pre-wrap;font-family:monospace;font-size:12px;color:#111'>"
        f"{html.escape(rollout['continuation'])}</div></div>"
    )


class ProblemBrowser:
    """Filter and browse problems and their rollouts. ``.selected`` is the current problem."""

    def __init__(self, problems: list):
        self.problems = problems
        self.selected = problems[0] if problems else None

        self.passrate_slider = widgets.FloatRangeSlider(
            value=(0.0, 1.0), min=0.0, max=1.0, step=0.01, description="passrate",
            continuous_update=False, layout=widgets.Layout(width="350px"),
        )
        self.search_box = widgets.Text(description="search", placeholder="filter question text",
                                       layout=widgets.Layout(width="350px"))
        self.problem_dropdown = widgets.Dropdown(description="problem", layout=widgets.Layout(width="750px"))
        self.rollout_slider = widgets.IntSlider(description="rollout", min=0, max=31,
                                                continuous_update=False,
                                                layout=widgets.Layout(width="350px"))
        self.correct_filter = widgets.Dropdown(
            description="show", options=["all rollouts", "only wrong", "only correct"],
            layout=widgets.Layout(width="250px"),
        )
        self.problem_out = widgets.HTML()
        self.rollout_out = widgets.HTML()

        self.passrate_slider.observe(self._refresh_options, names="value")
        self.search_box.observe(self._refresh_options, names="value")
        self.problem_dropdown.observe(self._on_select, names="value")
        self.rollout_slider.observe(self._show_rollout, names="value")
        self.correct_filter.observe(self._on_select, names="value")
        self._refresh_options()

    def _filtered(self):
        lo, hi = self.passrate_slider.value
        text = self.search_box.value.strip().lower()
        return [
            p for p in self.problems
            if lo <= p["pass_rate"] <= hi and (not text or text in p["question"].lower())
        ]

    def _refresh_options(self, _change=None):
        options = [
            (f"#{p['native_id']}  [{p['n_correct']}/{p['n_rollouts']}]  {p['question'][:90]}", p["native_id"])
            for p in self._filtered()
        ]
        self.problem_dropdown.options = options or [("no problems match the filters", None)]
        self.problem_dropdown.value = options[0][1] if options else None

    def _on_select(self, _change=None):
        native_id = self.problem_dropdown.value
        if native_id is None:
            self.selected = None
            self.problem_out.value = "<i>no problem selected</i>"
            self.rollout_out.value = ""
            return
        self.selected = next(p for p in self.problems if p["native_id"] == native_id)
        self.problem_out.value = _problem_html(self.selected)
        indices = self._rollout_indices()
        self.rollout_slider.max = max(len(indices) - 1, 0)
        self.rollout_slider.value = 0
        self._show_rollout()

    def _rollout_indices(self):
        if self.selected is None:
            return []
        mode = self.correct_filter.value
        return [
            i for i, r in enumerate(self.selected["rollouts"])
            if mode == "all rollouts"
            or (mode == "only wrong" and not r["correct"])
            or (mode == "only correct" and r["correct"])
        ]

    def _show_rollout(self, _change=None):
        indices = self._rollout_indices()
        if not indices:
            self.rollout_out.value = "<i>no rollouts match the filter</i>"
            return
        idx = indices[min(self.rollout_slider.value, len(indices) - 1)]
        self.rollout_out.value = _rollout_html(self.selected["rollouts"][idx], idx)

    def display(self):
        display(widgets.VBox([
            widgets.HBox([self.passrate_slider, self.search_box]),
            self.problem_dropdown,
            self.problem_out,
            widgets.HBox([self.rollout_slider, self.correct_filter]),
            self.rollout_out,
        ]))


class EditRunPanel:
    """Edit the selected problem (or write a new one), pick generation settings, run rollouts."""

    def __init__(self, runner, browser: ProblemBrowser = None, history_path: str = "rollout_runs/history.jsonl"):
        self.runner = runner
        self.browser = browser
        self.history_path = Path(history_path)
        self.baseline = None  # problem the edit started from, for comparison
        self.results = []

        self.question_box = widgets.Textarea(
            description="question", layout=widgets.Layout(width="900px", height="140px")
        )
        self.answer_box = widgets.Text(
            description="answer", placeholder="final numeric answer for the edited problem",
            layout=widgets.Layout(width="450px"),
        )
        self.n_rollouts = widgets.BoundedIntText(value=32, min=1, max=256, description="rollouts")
        self.temperature = widgets.BoundedFloatText(value=0.6, min=0.0, max=2.0, step=0.05, description="temp")
        self.top_p = widgets.BoundedFloatText(value=0.95, min=0.0, max=1.0, step=0.01, description="top_p")
        self.max_gen_toks = widgets.BoundedIntText(value=131072, min=16, max=131072, description="max toks")
        self.seed_box = widgets.Text(value="", description="seed", placeholder="blank = unseeded (original setup)")
        self.run_name = widgets.Text(value="", description="run name", placeholder="optional label")

        self.load_button = widgets.Button(description="Load selected problem", icon="download")
        self.run_button = widgets.Button(description="Run rollouts", button_style="primary", icon="play")
        self.status = widgets.HTML()
        self.output = widgets.Output()

        self.result_dropdown = widgets.Dropdown(description="run", layout=widgets.Layout(width="750px"))
        self.rollout_slider = widgets.IntSlider(description="rollout", min=0, max=0,
                                                continuous_update=False,
                                                layout=widgets.Layout(width="350px"))
        self.result_out = widgets.HTML()
        self.rollout_out = widgets.HTML()

        self.load_button.on_click(self._load_selected)
        self.run_button.on_click(self._run)
        self.result_dropdown.observe(self._show_result, names="value")
        self.rollout_slider.observe(self._show_result_rollout, names="value")

        if browser is not None and browser.selected is not None:
            self._load_selected()

    def _load_selected(self, _button=None):
        if self.browser is None or self.browser.selected is None:
            self.status.value = "<i>no problem selected in the browser</i>"
            return
        problem = self.browser.selected
        self.baseline = problem
        self.question_box.value = problem["question"]
        self.answer_box.value = str(problem["short_answer"])
        self.status.value = (
            f"loaded problem #{problem['native_id']} "
            f"(original {_passrate_badge(problem['pass_rate'], problem['n_correct'], problem['n_rollouts'])})"
        )

    def _run(self, _button=None):
        question = self.question_box.value.strip()
        answer = self.answer_box.value.strip()
        if not question or not answer:
            self.status.value = "<b style='color:#c62828'>question and answer are both required</b>"
            return
        seed = int(self.seed_box.value) if self.seed_box.value.strip() else None
        self.run_button.disabled = True
        self.status.value = "<b>running rollouts…</b> (progress below)"
        # NOTE: `with self.output:` (ipywidgets Output) captures exceptions and
        # renders them in the output area rather than propagating them, so we
        # must catch failures *inside* the context and bail out explicitly.
        with self.output:
            try:
                result = self.runner.run(
                    question=question,
                    answer=answer,
                    n_rollouts=self.n_rollouts.value,
                    temperature=self.temperature.value,
                    top_p=self.top_p.value,
                    max_gen_toks=self.max_gen_toks.value,
                    seed=seed,
                    run_name=self.run_name.value.strip() or None,
                )
            except Exception as exc:
                # Output's __exit__ swallows exceptions, so we can't rely on
                # `raise` reaching the caller; print the traceback into the
                # output area and stop the handler explicitly.
                self.status.value = (
                    f"<b style='color:#c62828'>run failed: {html.escape(str(exc))}</b>"
                )
                self.run_button.disabled = False
                traceback.print_exc()
                return
        result["baseline_native_id"] = self.baseline["native_id"] if self.baseline else None
        result["edited"] = self.baseline is None or question != self.baseline["question"]
        self.results.append(result)
        self._append_history(result)
        self.status.value = "<b style='color:#2e7d32'>done</b>"
        self.run_button.disabled = False

        labels = [
            (f"{i}: {Path(r['run_dir']).name}  [{r['n_correct']}/{r['n_rollouts']}]", i)
            for i, r in enumerate(self.results)
        ]
        self.result_dropdown.options = labels
        self.result_dropdown.value = labels[-1][1]

    def _append_history(self, result: dict):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in result.items() if k != "rollouts"}
        slim["rollouts"] = [
            {k: v for k, v in r.items() if k != "continuation"} for r in result["rollouts"]
        ]
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    def _show_result(self, _change=None):
        if self.result_dropdown.value is None or not self.results:
            return
        result = self.results[self.result_dropdown.value]
        comparison = ""
        if self.baseline is not None and result.get("baseline_native_id") == self.baseline["native_id"]:
            comparison = (
                f"<div style='margin-top:6px'>original problem #{self.baseline['native_id']}: "
                f"{_passrate_badge(self.baseline['pass_rate'], self.baseline['n_correct'], self.baseline['n_rollouts'])}"
                f" &nbsp;→&nbsp; this run: "
                f"{_passrate_badge(result['pass_rate'], result['n_correct'], result['n_rollouts'])}</div>"
            )
        settings = html.escape(json.dumps(result["generation_settings"]))
        self.result_out.value = (
            _problem_html(result)
            + comparison
            + f"<div style='margin-top:4px;color:#555'><small>settings: {settings}</small></div>"
        )
        self.rollout_slider.max = max(result["n_rollouts"] - 1, 0)
        self.rollout_slider.value = 0
        self._show_result_rollout()

    def _show_result_rollout(self, _change=None):
        if self.result_dropdown.value is None or not self.results:
            return
        result = self.results[self.result_dropdown.value]
        idx = min(self.rollout_slider.value, result["n_rollouts"] - 1)
        self.rollout_out.value = _rollout_html(result["rollouts"][idx], idx)

    def display(self):
        display(widgets.VBox([
            widgets.HBox([self.load_button, self.status]),
            self.question_box,
            self.answer_box,
            widgets.HBox([self.n_rollouts, self.temperature, self.top_p]),
            widgets.HBox([self.max_gen_toks, self.seed_box, self.run_name]),
            self.run_button,
            self.output,
            self.result_dropdown,
            self.result_out,
            self.rollout_slider,
            self.rollout_out,
        ]))
