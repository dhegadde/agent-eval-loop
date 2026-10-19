# Experimental log

Append a new section per run. Always record the baseline — without it you cannot
tell whether optimization helped.

---

## Run 1 — Gemini, token-overlap scorer (2026-08-31)

**Config:** `gemini-3.1-flash-lite` executor, `gemini-3.6-flash` optimizer,
8 steps, batch 2, 4 train / 2 dev / 2 test. Reply scored by token overlap.

| Step | score | category_acc | priority_score | reply_score |
|---|---|---|---|---|
| **baseline** | **0.7320** | 1.00 | 1.00 | 0.330 |
| 2 | 0.6698 | 1.00 | 0.75 | 0.300 |
| 4 | 0.6949 | 1.00 | 0.75 | 0.362 |
| 6 | 0.6884 | 1.00 | 0.75 | 0.346 |

**Nothing beat baseline.** Rollback fired at step 6 and correctly reverted to the
step-4 snapshot. Runtime ~26 min, crashed at step 7.

### Findings

**TextGrad damaged a prompt that was already perfect.** `priority_score` dropped
1.00 → 0.75 at step 2 and never recovered. It then spent five steps optimizing
around self-inflicted damage.

**The scorer rewarded invention.** Baseline replies were well grounded — the
billing one asked for statement descriptors, authorization numbers, pending-vs-
posted status, all traceable to the ticket. By step 6 the draft agent was writing:

> "For security, we won't ask for full card details—your card ending 4417 is
> sufficient for verification."

A security policy that appears nowhere in the ticket or the reference. Token
overlap scores fluent invention well, so optimization walked straight toward it.
This is the single strongest argument for the LLM judge.

**Crash:** TextGrad rewrote the draft instruction, dropped the `final_category`
passthrough wording, pydantic rejected the output, node returned `None`. See
GOTCHAS #7.

### Conclusion

Working as designed, measuring the wrong thing, on far too little data. Both
causes addressed in the next revision; neither has been validated yet.

---

## Run 2 — OpenAI, LLM judge (attempted 2026-09-04)

Crashed during the baseline eval: `TypeError: unsupported operand type(s) for +:
'int' and 'str'` — a critique string in the metrics dict (GOTCHAS #8). Fixed by
routing critiques to `judge_critiques.jsonl`.

Judge calls themselves worked (visible as +$0.002 per example). Baseline replies
again looked well grounded.

**No scores recorded. The judge has still never completed a full run.**

---

## Judge validation — OpenAI `gpt-5.4` judge (2026-09-04)

ROADMAP #2. Not a full optimization run — the judge alone, exercised through the
real `TicketBenchmark._score_reply` path against six curated cases with known
correct verdicts (`validate_judge.py`). Six judge calls, **~$0.016 total**.

| case | expect | hallucinated | score | verdict |
|---|---|---|---|---|
| grounded_refund | clean | False | 0.97 | PASS |
| invented_security_policy | flag | True | 0.00 | PASS |
| invented_timeline | flag | True | 0.00 | PASS |
| grounded_login | clean | False | 0.98 | PASS |
| restates_only | clean | False | 0.29 | PASS |
| invented_refund_amount | flag | True | 0.00 | PASS |

### Findings

- **The gate works.** It caught all three inventions, including Run 1's canonical
  invented security policy ("card ending 4417 is sufficient for verification") —
  the case ROADMAP #2 says it *must* catch. It fires on `hallucinated`; never
  always-0.0.
- **No false positives on grounded replies** — after a rubric fix. First pass,
  `grounded_login` was wrongly flagged (score 0.0): the judge penalized it for
  stating a root cause and an action that the *reference reply* asserts but the
  *ticket* alone does not. That is exactly the "rubric too strict" failure ROADMAP
  #2 warns about — it would have flagged the approved answer itself, and pushed
  optimization toward uselessly hedged replies.
- **Rubric fix:** `JUDGE_PROMPT` now grounds a claim against the ticket **OR** the
  reference reply, and flags `hallucinated` only for inventions supported by
  neither. Re-run: 6/6, `grounded_login` 0.0 → 0.98, all three flag cases still
  fire.
- **Discrimination is real:** grounded 0.97/0.98 vs content-free "restates the
  problem" 0.29 (not flagged, just low). The judge separates grounded-but-weak
  from ungrounded.

**Conclusion:** the judge is validated on crafted cases and the hallucination gate
is live and correctly calibrated. It has still never run over the *dataset* inside
a full optimization loop — that remains gated on ROADMAP #1 (50+ real tickets),
since a real run on 8 examples is noise by the project's own standard.

---

## Run 3 — OpenAI, validated LLM judge (2026-09-04)

**Config:** `gpt-5.4-nano` executor, separate `gpt-5.4` optimizer and judge,
8 steps, batch 2, 4 train / 2 dev / 2 test, smoothing 1.0. Runtime 46 minutes;
logged cost approximately **$4.814**.

| Point | raw_score | category_acc | priority_score | reply_score | hallucinated |
|---|---:|---:|---:|---:|---:|
| **baseline dev** | **0.752** | 1.00 | 1.00 | 0.380 | 0.00 |
| step 2 dev | 0.828 | 1.00 | 1.00 | 0.570 | 0.00 |
| **step 4 dev (best)** | **0.856** | 1.00 | 1.00 | 0.640 | 0.00 |
| step 6 dev | 0.834 | 1.00 | 1.00 | 0.585 | 0.00 |
| step 8 dev | 0.696 | 1.00 | 1.00 | 0.240 | 0.00 |
| **best graph, held-out test** | **0.422** | 0.50 | 0.75 | 0.180 | 0.00 |

### Findings

- The full optimization loop and live judge completed for the first time.
- Rollback worked: step 8 regressed and the saved final/best graph is the step-4
  snapshot.
- The judge avoided Run 1's invented-policy failure on recorded dev checks.
- The dev gain did **not** generalize. The held-out result was substantially worse,
  confirming that two dev examples are not a usable selection signal.
- The optimizer ignored its prose-only 150-word constraint. The saved best graph's
  three instructions are approximately 1,600 / 1,228 / 1,011 words. A real
  structural gate was subsequently added to `evolve.py`.

### Conclusion

Mechanically successful, scientifically inconclusive, and clear evidence of
overfitting. Do not run another autonomous optimization on this 8-row dataset.

---

## Human-gated evolution — ticket gen1 (2026-09-04)

One proposal was shown to the user with its complete prompt diff and score delta.
The user accepted it despite a documented passthrough/bloat risk. It is preserved
as `optimized/ticket_triage_gen1_accepted.json`; the accepted pointer is
`ticket_triage_evolved_current.json`.

This was a successful test of governance, not evidence of model quality: acceptance
created an offline, versioned, reversible artifact and changed no serving path.

A later second proposal remains unresolved in `optimized/_pending/`. Its recorded
dev raw score regresses 0.736 → 0.684, both sides have hallucinated=0.5, and the
candidate exceeds the word limit on every agent. Do not decide it automatically.

---

## Template for the next run

```
## Run N — <config> (<date>)

**Config:** executor / optimizer / judge models, steps, batch, split sizes,
smoothing on|off, dataset size.

| Step | score | raw_score | category_acc | priority_score | reply_score | hallucinated |

### Findings
- Did anything beat baseline?
- What did the rewritten instructions actually change? (read optimized/*.json)
- What did the judge object to? (judge_critiques.jsonl)
- Any hallucination gate fires? Were they real?
```

## Things to check every run

- **Baseline first.** Written down before anything else.
- **`raw_score` vs `score`.** With `smoothing=1.0` they differ; ranking is what
  the optimizer uses, but report raw for cross-run comparison.
- **`hallucinated` rate.** If it's always 0.0 the gate isn't working. If it fires
  on grounded replies the rubric is too strict and you'll optimize toward
  uselessly hedged text.
- **The rewritten instructions**, not just the numbers. On a small dataset expect
  overfitting — something like "billing tickets about duplicate charges are P1" is
  memorization of a training row, not a learned rule.
