# Agent Eval Loop

A three-agent support-ticket pipeline built on [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX), plus the part that
actually mattered: a benchmark, a validated LLM judge, prompt optimization with rollback, and a human approval gate.
The honest headline is in the results: optimization looked like it worked on dev data and did not hold up on held-out
data, and the first scorer rewarded the model for making things up.

Dheeraj Gadde, 2026. Self-directed project; small dataset, so read the results as engineering evidence, not model
quality claims.

## What's here

**The pipeline** (`setup_swarm.py`, `ticket_task.py`): three agents in a graph, `triage` (category + priority) →
`extract_facts` → `draft_reply`, passing structured JSON between stages against declared output schemas.

**The benchmark** (`benchmark_tickets.py`, `raw_evaluator.py`): seeded train / dev / test splits over
`data/tickets.jsonl` (8 synthetic tickets with approved replies). It scores category accuracy, priority and reply
quality on the pipeline's raw outputs, not on an LLM summary of them.

**The LLM judge** (`validate_judge.py`): grades each reply for grounding and flags hallucinations. A reply that invents
anything not supported by the ticket or the approved reference reply gets a score of 0.

**Optimization with guardrails** (`optimize_textgrad.py`, `evolve.py`): TextGrad rewrites prompts; every step is
snapshotted (`optimized/`) and rolled back on regression. `evolve.py` is the human-gated version: it proposes one
rewrite at a time, shows the full prompt diff and score delta, and changes nothing until a person accepts it.

**Reusable jobs** (`jobs.py`, `job_specs.py`, `generic_task.py`, `generic_benchmark.py`): point the same gated loop at
any single-input, single-output text job defined as data (`job.json` plus examples), not code.

**Provider-neutral models** (`model_runtime.py`): executor, optimizer and judge are configured independently (OpenAI or
any OpenAI-compatible endpoint, such as a self-hosted vLLM server).

## Results (full log in [`docs/RESULTS.md`](docs/RESULTS.md))

**Run 1: the scorer rewarded invention.** With a token-overlap scorer, nothing beat the baseline, and by step 6 the
reply agent had invented a security policy that appears nowhere in the ticket ("your card ending 4417 is sufficient
for verification"). Fluent invention scores well on token overlap, so optimization walked toward it. That is why the
judge exists.

**Judge validation: 6/6 curated cases, about $0.016.** It caught all three planted hallucinations (invented security
policy, invented timeline, invented refund amount) with no false positives on grounded replies, after one rubric fix:
the first version flagged an approved reply for stating facts that were in the reference but not the ticket.

| Case | Expected | Judge score | Verdict |
|---|---|---:|---|
| grounded_refund | clean | 0.97 | pass |
| invented_security_policy | flag | 0.00 | pass |
| invented_timeline | flag | 0.00 | pass |
| grounded_login | clean | 0.98 | pass |
| restates_only | clean (weak) | 0.29 | pass |
| invented_refund_amount | flag | 0.00 | pass |

**Run 3: dev gains did not generalize.** With the validated judge, 8 optimization steps, 46 minutes, about $4.81:

| Point | Score |
|---|---:|
| Baseline (dev) | 0.752 |
| Best step 4 (dev) | 0.856 |
| Step 8 (dev, rolled back) | 0.696 |
| **Best graph on held-out test** | **0.422** |

Rollback worked and the judge kept hallucinations at zero, but two dev examples are not a usable selection signal, so
this is reported as overfitting. The optimizer also ignored a prose-only 150-word limit (best instructions ran to
1,000-1,600 words), so a real structural gate was added to `evolve.py`.

**Human-gated evolution:** one proposal was reviewed and accepted with its full diff (versioned and reversible); a
second, which regressed 0.736 to 0.684 and broke the word limit, is left undecided on purpose.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install evoagentx openai python-dotenv jsonschema pytest
python dry_run.py              # validates the whole pipeline with zero API calls
pytest -q                      # zero-network tests for the gate, job layer and model config
cp .env.example .env           # add a key only when you want a paid run
python validate_judge.py       # ~6 judge calls
python evolve.py               # human-gated optimization
```

EvoAgentX 0.1.4 has several sharp edges; [`docs/GOTCHAS.md`](docs/GOTCHAS.md) lists each one found while building this
(each cost a failed run).

## Limits

- 8 tickets is far too few for optimization; the honest next step is 50+ real, labeled tickets.
- The judge is validated on 6 crafted cases, not a large labeled set.
- Costs are from logged usage, not invoices.

## License

MIT.
