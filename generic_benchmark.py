"""Benchmark for any single-input, single-output text job."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from evoagentx.benchmark.benchmark import Benchmark

from job_specs import data_path_for, load_examples


JUDGE_TEMPLATE = """You evaluate candidate output for a user-defined text job.

JOB:
{goal}

SOURCE INPUT:
{source}

HUMAN-WRITTEN REFERENCE (one acceptable answer, not necessarily the only one):
{reference}

CANDIDATE OUTPUT:
{candidate}

QUALITY RUBRIC:
{rubric}

HARD RULES:
{hard_rules}

Score quality from 0.0 to 1.0 using the rubric. Set hard_failure=true only if the
candidate violates an explicit HARD RULE above. Do not invent additional hard
rules. If there are no hard rules, hard_failure must be false. The reference is a
guide to quality and facts, not a demand for matching wording.

Return ONLY valid JSON:
{{"hard_failure": true or false,
  "score": 0.0 to 1.0,
  "critique": "one sentence naming the single biggest weakness"}}"""


class GenericTextBenchmark(Benchmark):
    def __init__(self, spec_path: Path, spec: dict, judge_llm: Any = None,
                 smoothing: float = 1.0, **kwargs):
        self.spec_path = spec_path
        self.spec = spec
        self.judge_llm = judge_llm
        self.smoothing = smoothing
        self.output_name = spec["output"]["name"]
        self.evaluation = spec["evaluation"]
        self._judge_cache: dict[tuple[str, ...], tuple[float, bool, str, bool]] = {}
        self._cache_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self.diagnostics_path = spec_path.parent / "critiques.jsonl"
        super().__init__(
            name=spec["name"],
            path=str(data_path_for(spec_path, spec)),
            mode="all",
            **kwargs,
        )

    def _load_data(self):
        rows = load_examples(self.spec_path, self.spec)
        self._train_data = [row for row in rows if row["split"] == "train"]
        self._dev_data = [row for row in rows if row["split"] == "dev"]
        self._test_data = [row for row in rows if row["split"] == "test"]
        if len(self._train_data) < 2 or len(self._dev_data) < 2:
            raise ValueError(
                "A job needs at least 2 train and 2 dev examples before evolution. "
                "Run `python jobs.py status <job.json>`."
            )

    def _get_id(self, example: Any) -> Any:
        return example["id"]

    def _get_label(self, example: Any) -> str:
        # TextGrad requires labels to be strings for non-coding benchmarks.
        return json.dumps({
            "id": example["id"],
            "input": example["input"],
            "reference": example["reference"],
        })

    def evaluate(self, prediction: Any, label: Any) -> dict[str, float]:
        gold = json.loads(label) if isinstance(label, str) else label
        candidate = self._candidate(prediction)

        if self.evaluation["mode"] == "exact":
            case_sensitive = bool(self.evaluation.get("case_sensitive", False))
            expected = self._normalize(gold["reference"], case_sensitive)
            actual = self._normalize(candidate, case_sensitive)
            quality = float(actual == expected)
            hard_failure = False
            judge_error = False
            critique = "Exact match." if quality else "Candidate does not match the reference."
        else:
            quality, hard_failure, critique, judge_error = self._judge(
                source=gold["input"], reference=gold["reference"], candidate=candidate
            )

        raw = 0.0 if hard_failure or judge_error else quality
        self._log_diagnostic(gold["id"], quality, hard_failure, judge_error, critique)
        return {
            "quality_score": float(quality),
            "hard_failure": float(hard_failure),
            "judge_error": float(judge_error),
            "score": _beta_smooth(raw, self.smoothing),
            "raw_score": float(raw),
        }

    def _candidate(self, prediction: Any) -> str:
        if isinstance(prediction, dict):
            value = prediction.get(self.output_name, "")
            return value if isinstance(value, str) else str(value)
        text = str(prediction)
        block = re.search(r"\{.*\}", text, re.S)
        if block:
            try:
                value = json.loads(block.group(0)).get(self.output_name, "")
                return value if isinstance(value, str) else str(value)
            except json.JSONDecodeError:
                pass
        return text

    @staticmethod
    def _normalize(text: str, case_sensitive: bool) -> str:
        normalized = " ".join(text.split())
        return normalized if case_sensitive else normalized.casefold()

    def _judge(self, source: str, reference: str, candidate: str):
        if not candidate.strip():
            return 0.0, False, "Empty output.", False
        if self.judge_llm is None:
            return 0.0, False, "Judge unavailable; refusing to guess a score.", True

        hard_rules = self.evaluation.get("hard_rules", [])
        rules_text = "\n".join(f"- {rule}" for rule in hard_rules) or "(none)"
        prompt = JUDGE_TEMPLATE.format(
            goal=self.spec["goal"], source=source, reference=reference,
            candidate=candidate, rubric=self.evaluation["rubric"],
            hard_rules=rules_text,
        )
        key = (source[:500], reference[:500], candidate[:1000], prompt[-1000:])
        with self._cache_lock:
            cached = self._judge_cache.get(key)
        if cached is not None:
            return cached

        try:
            raw = self.judge_llm.generate(prompt=prompt)
            text = getattr(raw, "content", None) or str(raw)
            block = re.search(r"\{.*\}", text, re.S)
            data = json.loads(block.group(0)) if block else {}
            # With no declared hard rules, the judge has no authority to invent one.
            hard_failure = bool(data.get("hard_failure", False)) if hard_rules else False
            quality = max(0.0, min(1.0, float(data.get("score", 0.0))))
            critique = str(data.get("critique", ""))[:400]
            result = (quality, hard_failure, critique, False)
        except Exception as exc:
            # Fail closed. A broken judge must not create a flattering fallback score.
            result = (0.0, False, f"judge error: {exc}"[:400], True)

        with self._cache_lock:
            self._judge_cache[key] = result
        return result

    def _log_diagnostic(self, example_id: str, quality: float, hard_failure: bool,
                        judge_error: bool, critique: str) -> None:
        row = {
            "id": example_id,
            "quality_score": round(float(quality), 3),
            "hard_failure": bool(hard_failure),
            "judge_error": bool(judge_error),
            "critique": critique,
        }
        try:
            with self._log_lock, self.diagnostics_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass


def _beta_smooth(score: float, amount: float) -> float:
    if amount <= 0:
        return score
    return (score + amount) / (1.0 + 2.0 * amount)
