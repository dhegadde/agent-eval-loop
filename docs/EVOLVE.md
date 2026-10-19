# evolve.py — self-improvement with a human gate

The offline optimizer from `optimize_textgrad.py` runs all 8 steps on its own and
auto-keeps whatever scores highest. `evolve.py` is the same machinery with **one
change: you approve every evolution before it is kept.** It proposes a single
prompt rewrite, shows you exactly what changed and what that did to the dev score,
and waits. Reject, and the graph snaps back to your last accepted version.

This is the point Ivan missed in the story we keep coming back to — the human in
the loop stays a real decision, not a rubber stamp. It is also just good
engineering: Run 1's invented security policy was *auto-accepted*. Here it
wouldn't be, because you'd see it proposed and say no.

## Run it

```bash
python evolve.py                 # default task: support tickets
python evolve.py --max-steps 12  # allow up to 12 proposals
python evolve.py --smoothing 0   # once you have 50+ examples (ROADMAP #3)
python evolve.py --job jobs/my_job/job.json --propose-only
```

Each proposal costs **one optimizer call + one dev eval** — you only pay for
evolutions you let it propose. For every proposal you get:

- the **before → after** of each instruction it rewrote (unchanged agents are noted, not dumped);
- the **dev score delta**, per metric, with `hallucinated` flagged as "higher is worse";
- a prompt: `[a]ccept  [r]eject  [q]uit`.

Every accepted generation is saved as its own file — `optimized/<task>_gen<N>_accepted.json`
— so you can always roll back to a specific one. The graph you'd actually serve is
written to `optimized/<task>_evolved_current.json` and is *only ever* a version you
accepted. Nothing touches the runtime path; that stays frozen (CLAUDE.md, "The two loops").

Only one proposal can be pending at a time. A second `--propose-only` run is
refused until you accept or reject the first. Instruction length is enforced in
code (150 words by default); ordinary acceptance is blocked on a violation. A
deliberate `--override-constraints` flag exists for final human authority.

## Point it at any job

`evolve.py` itself is task-agnostic. A "job" is a module exposing the contract in
`ticket_task.py`. To add one, `python evolve.py --task my_task` where `my_task.py`
provides:

| name | is | 
|---|---|
| `build_graph()` | a `SequentialWorkFlowGraph` whose task `prompt_template.instruction`s are what gets rewritten |
| `make_benchmark(judge, smoothing)` | a `Benchmark`; `evaluate(pred, label)` returns a **numbers-only** metrics dict (send any text to a side file — GOTCHAS #8) |
| `collate(example)` | one data row → `{graph_input_name: value}` |
| `executor_config`, `executor_llm`, `optimizer_llm`, `judge_llm` | the three LLM roles |

For the common single-input/single-output case, no Python module is needed.
Create a portable job with `jobs.py`, then run:

```bash
python evolve.py --job jobs/<name>/job.json --propose-only
```

See `JOBS.md` for creation, real-example collection, validation, and test freezing.

### What every new job needs (there is no way around this)

The optimizer improves a prompt *against a score*. So for any job you must supply:

1. **Examples** — a JSONL of real input rows, each with a reference/gold output.
   ~50+ for a trustworthy signal (ROADMAP #1); fewer is noise you can still watch.
2. **A scoring function** — how right is an output? For open-ended text, reuse the
   validated LLM judge pattern in `benchmark_tickets.py` (`JUDGE_PROMPT` +
   `_score_reply`), rewriting the rubric for your task. For exact-answer tasks
   (classification, extraction), a direct comparison is better and free.
3. **A seed instruction** per agent — the starting prompt TextGrad edits.

A job with no scorable outcome (e.g. "pick winning trades") can't be optimized
here — there's nothing to climb. That's a feature: it's the same guardrail that
keeps this from optimizing toward confident nonsense.

### The simplest job: one agent

The ticket task is a 3-agent pipeline, but a job can be a single agent —
`build_graph()` returning a one-task graph (input → output), scored by the judge.
That is the natural shape for "summarize this", "draft a reply to this",
"rewrite this in my voice": one instruction, evolved against your labeled
examples, one approval at a time.
