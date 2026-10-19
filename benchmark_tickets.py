"""Your eval set + the fitness function. This file is the actual project.

Three things changed from the placeholder version:

1. LLM JUDGE replaces token overlap. The old `_reply_heuristic` scored vocabulary
   reuse, not correctness - a reply could be factually wrong and score well by
   echoing the reference's words. 40% of the composite was measuring nothing.

2. HALLUCINATION HARD GATE. A reply asserting anything the ticket doesn't support
   scores 0 on the reply component regardless of fluency. Without this, optimization
   reliably discovers that confident invention scores well with a soft judge.

3. BETA(1,1) SMOOTHING on the aggregate. With a handful of dev examples a raw mean
   is unstable - one flip moves it 50 points, and the optimizer chases noise.
   Smoothing pulls small-sample estimates toward 0.5 until evidence accumulates.
   Set EvalConfig.smoothing=0 to turn it off once you have 50+ examples.
"""

import json
import os
import random
import re
import threading
from pathlib import Path
from typing import Any, Optional

from evoagentx.benchmark.benchmark import Benchmark

PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
VALID_CATEGORIES = {"billing", "bug", "account_access", "feature_request", "other"}

W_CATEGORY, W_PRIORITY, W_REPLY = 0.4, 0.2, 0.4

# Text diagnostics are written here, NOT returned in the metrics dict. The stock
# Evaluator (evaluator.py:327) sums every key across examples; a string critique
# in that dict raises "unsupported operand type(s) for +: 'int' and 'str'"
# (GOTCHAS #8). Delete this file between runs - it appends.
CRITIQUES_PATH = os.environ.get("JUDGE_CRITIQUES_PATH", "judge_critiques.jsonl")
_CRITIQUE_LOCK = threading.Lock()

JUDGE_PROMPT = """You grade drafted customer-support replies. Be strict.

TICKET (what the customer told us):
{ticket}

REFERENCE REPLY (an approved answer written by an agent with account access; one
acceptable answer, not the only one):
{reference}

CANDIDATE REPLY:
{candidate}

Grade the candidate on:
- Factual grounding. A claim is grounded if it is supported by the ticket OR by
  the reference reply. The reference is an approved answer, so any diagnosis or
  action it states - identifying a root cause, clearing a session, issuing a
  refund - is legitimate, and a candidate that mirrors those facts is NOT
  hallucinating. A HALLUCINATION is a claim supported by NEITHER source: an
  invented refund amount, timeline, root cause, policy, or account detail that
  appears in neither, or a claim that contradicts them. Set hallucinated=true
  only for such unsupported inventions - not merely for stating a fact the
  reference also states. Matching the reference's facts matters; matching its
  wording does not.
- Actionability: does it move the ticket forward, or just restate the problem?
- Tone and length: direct and professional. Filler openings and padding are faults.

Return ONLY valid JSON, no prose and no code fences:
{{"hallucinated": true or false,
  "score": 0.0 to 1.0,
  "critique": "one sentence naming the single biggest weakness"}}"""


class TicketBenchmark(Benchmark):
    """Support-ticket triage + reply drafting.

    Splits one JSONL file into train/dev/test. Keep test untouched until the end.
    """

    def __init__(self, path: str = "data/tickets.jsonl", mode: str = "all",
                 seed: int = 0, judge_llm: Any = None, smoothing: float = 1.0, **kwargs):
        self.seed = seed
        self.judge_llm = judge_llm      # None -> falls back to the crude heuristic
        self.smoothing = smoothing      # Beta(a,a) pseudo-counts; 0 disables
        self._judge_cache: dict = {}
        self._cache_lock = threading.Lock()
        super().__init__(name="ticket_triage", path=path, mode=mode, **kwargs)

    def _load_data(self):
        rows = [
            json.loads(line)
            for line in Path(self.path).read_text().splitlines()
            if line.strip()
        ]
        for i, r in enumerate(rows):
            r.setdefault("id", f"t{i}")
        random.Random(self.seed).shuffle(rows)

        n = len(rows)
        n_train = max(1, int(n * 0.5))
        n_dev = max(1, int(n * 0.25))

        if self.mode in ("all", "train"):
            self._train_data = rows[:n_train]
        if self.mode in ("all", "dev"):
            self._dev_data = rows[n_train:n_train + n_dev]
        if self.mode in ("all", "test"):
            self._test_data = rows[n_train + n_dev:]

    def _get_id(self, example: Any) -> Any:
        return example["id"]

    def _get_label(self, example: Any) -> Any:
        """MUST return a string.

        TextGradOptimizer.step() raises "Label must be a string for non-coding
        benchmarks" if this returns a dict. So we serialize, and evaluate()
        deserializes on the way back in.
        """
        return json.dumps({
            "ticket": example["ticket"],
            "category": example["category"],
            "priority": example["priority"],
            "reply": example["reply"],
        })

    def evaluate(self, prediction: Any, label: Any) -> dict:
        gold = _as_label_dict(label)
        pred = _coerce(prediction)

        cat_ok = pred.get("category", "").strip().lower() == gold["category"]

        p_pred = pred.get("priority", "").strip().upper()
        pri_exact = p_pred == gold["priority"]
        pri_near = abs(
            PRIORITY_ORDER.get(p_pred, 9) - PRIORITY_ORDER.get(gold["priority"], 9)
        ) <= 1
        pri_score = 1.0 if pri_exact else (0.5 if pri_near else 0.0)

        reply = (pred.get("reply") or "").strip()
        reply_score, hallucinated, critique = self._score_reply(
            ticket=gold.get("ticket", ""),
            reference=gold["reply"],
            candidate=reply,
        )

        raw = W_CATEGORY * float(cat_ok) + W_PRIORITY * pri_score + W_REPLY * reply_score

        # Critique is a string -> side-channel, never the metrics dict (GOTCHAS #8).
        self._log_critique(gold, pri_score, reply_score, hallucinated, critique)

        return {
            "category_acc": float(cat_ok),
            "priority_score": pri_score,
            "reply_score": reply_score,
            "hallucinated": float(hallucinated),
            "score": _beta_smooth(raw, self.smoothing),
            "raw_score": raw,
        }

    def _log_critique(self, gold: dict, pri_score: float, reply_score: float,
                      hallucinated: bool, critique: str) -> None:
        """Append one judge diagnostic to CRITIQUES_PATH. Thread-safe (num_workers>1)."""
        row = {
            "gold_category": gold.get("category", ""),
            "gold_priority": gold.get("priority", ""),
            "priority_score": round(float(pri_score), 3),
            "reply_score": round(float(reply_score), 3),
            "hallucinated": bool(hallucinated),
            "critique": critique,
        }
        try:
            with _CRITIQUE_LOCK, open(CRITIQUES_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass  # a diagnostics-logging failure must never kill an eval

    def _score_reply(self, ticket: str, reference: str, candidate: str):
        """-> (score, hallucinated, critique)."""
        if not candidate:
            return 0.0, False, "Empty reply."

        if self.judge_llm is None:
            return _reply_heuristic(candidate, reference), False, "heuristic (no judge)"

        key = (ticket[:200], candidate[:400])
        with self._cache_lock:
            if key in self._judge_cache:
                return self._judge_cache[key]

        prompt = JUDGE_PROMPT.format(
            ticket=ticket, reference=reference, candidate=candidate
        )
        try:
            raw = self.judge_llm.generate(prompt=prompt)
            text = getattr(raw, "content", None) or str(raw)
            block = re.search(r"\{.*\}", text, re.S)
            data = json.loads(block.group(0)) if block else {}
            hallucinated = bool(data.get("hallucinated", False))
            score = 0.0 if hallucinated else max(0.0, min(1.0, float(data.get("score", 0.0))))
            critique = str(data.get("critique", ""))[:200]
            result = (score, hallucinated, critique)
        except Exception as e:                        # judge failure must not kill a run
            result = (_reply_heuristic(candidate, reference), False, f"judge error: {e}")

        with self._cache_lock:
            self._judge_cache[key] = result
        return result


def _beta_smooth(score: float, a: float) -> float:
    """Beta(a, a) posterior mean for a single observation.

    Pulls an individual score toward 0.5 by `a` pseudo-counts. With a=1 a raw 1.0
    becomes 0.667 and a raw 0.0 becomes 0.333, so a single lucky example can no
    longer carry a candidate. Averaged over a real eval set the shrinkage washes
    out; on 2 examples it is exactly the stabilizer you want.
    """
    if a <= 0:
        return score
    return (score + a) / (1.0 + 2.0 * a)


def _as_label_dict(label: Any) -> dict:
    """Labels travel as JSON strings (TextGrad requires it). Accept either form."""
    if isinstance(label, dict):
        return label
    try:
        return json.loads(label)
    except (json.JSONDecodeError, TypeError):
        return {"ticket": "", "category": "", "priority": "", "reply": str(label)}


def _coerce(prediction: Any) -> dict:
    """Workflow output -> the three graded fields.

    RawOutputEvaluator hands us every node's outputs, so `category` and `priority`
    come straight off the triage node. The `final_` spellings are still accepted
    for older saved graphs.
    """
    raw = prediction if isinstance(prediction, dict) else None
    if raw is None:
        text = str(prediction)
        block = re.search(r"\{.*\}", text, re.S)
        if block:
            try:
                raw = json.loads(block.group(0))
            except json.JSONDecodeError:
                raw = None
        if raw is None:
            raw = {}
            for key in ("final_category", "final_priority", "category", "priority", "reply"):
                m = re.search(rf'"?{key}"?\s*[:=]\s*"?([^"\n,}}]+)', text, re.I)
                if m:
                    raw[key] = m.group(1).strip()

    out = {k: v for k, v in raw.items()}
    if "category" not in out and "final_category" in out:
        out["category"] = out["final_category"]
    if "priority" not in out and "final_priority" in out:
        out["priority"] = out["final_priority"]
    for k in ("category", "priority", "reply"):
        if not isinstance(out.get(k), str):
            out[k] = "" if out.get(k) is None else str(out[k])
    return out


def _reply_heuristic(candidate: str, reference: str) -> float:
    """Fallback only, used when no judge_llm is supplied or the judge errors.

    Token overlap. Rewards length and vocabulary, not correctness. Do not
    optimize against this.
    """
    if not candidate.strip():
        return 0.0
    c = set(re.findall(r"\w+", candidate.lower()))
    r = set(re.findall(r"\w+", reference.lower()))
    if not r:
        return 0.0
    overlap = len(c & r) / len(r)
    return min(1.0, overlap) * (1.0 if len(candidate) < 900 else 0.7)