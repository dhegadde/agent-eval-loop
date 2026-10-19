"""Human-gated evolution loop.

The optimizer proposes ONE prompt rewrite at a time, then stops and shows you:
  - the exact before -> after of every instruction it changed, and
  - what that change did to the score on the dev set.
Nothing is committed until you approve it. Reject and it is discarded and the
graph snaps back to your last accepted version. This is the offline optimization
loop from CLAUDE.md's "two loops", with a human gate on every evolution instead
of automatic score-based rollback.

Why a gate at all: automatic acceptance is how a prompt drifts somewhere you
never chose (the invented security policy in RESULTS.md Run 1 was auto-accepted).
Here every generation is a decision you make, on evidence you can see, and is
saved as its own versioned file so you can always roll back to it.

    python evolve.py                 # default task (support tickets)
    python evolve.py --max-steps 12
    python evolve.py --smoothing 0   # once you have 50+ examples

Runtime is untouched: the graph you actually serve is only ever a version you
accepted and saved. This script writes to ./optimized/ and spends money only on
the evolutions you let it propose (one optimizer + one dev eval per proposal).

TASK-AGNOSTIC by design: everything specific to the ticket task lives behind the
TASK object below. Point --task at another module exposing the same fields to run
this on any text task (see docs/EVOLVE.md).
"""

import argparse
import importlib
import json
import textwrap
from copy import deepcopy
from pathlib import Path

# MUST precede any LiteLLM construction (monkeypatches litellm.completion).
from rate_limit import install_rpm_limit

install_rpm_limit(max_rpm=200)

from evoagentx.agents import AgentManager  # noqa: E402
from evoagentx.optimizers.textgrad_optimizer import TextGradOptimizer  # noqa: E402
from evoagentx.workflow import WorkFlowGraph  # noqa: E402

from raw_evaluator import RawOutputEvaluator  # noqa: E402

# Metrics shown per proposal. `hallucinated` is the one where UP is bad.
DEFAULT_SHOWN = ["raw_score", "category_acc", "priority_score", "reply_score", "hallucinated"]
DEFAULT_WORSE_WHEN_UP = {"hallucinated"}


def load_task(name: str, job: str | None = None):
    """Import a task module. It must expose: build_graph(), make_benchmark(judge, smoothing),
    collate(example), and executor_config / executor_llm / optimizer_llm / judge_llm."""
    task = importlib.import_module(name)
    if job is not None:
        configure = getattr(task, "configure", None)
        if configure is None:
            raise ValueError(f"Task module {name!r} does not support --job.")
        configure(job)
    return task


def metric_config(task) -> tuple[list[str], set[str]]:
    """Let each task choose the metrics displayed at the human gate."""
    shown = list(getattr(task, "shown_metrics", DEFAULT_SHOWN))
    worse = set(getattr(task, "worse_when_up", DEFAULT_WORSE_WHEN_UP))
    return shown, worse


def instructions(graph: WorkFlowGraph) -> dict:
    """{agent_name: instruction} - the only thing TextGrad rewrites."""
    return {t["name"]: t["prompt_template"]["instruction"]
            for t in graph.get_config()["tasks"]}


def data_iterator(benchmark, collate, batch_size, seed=0):
    """Mirrors TextGradOptimizer.optimize()'s internal batching, one batch per proposal."""
    epoch = 0
    while True:
        eseed = seed + epoch if seed is not None else None
        train = benchmark.get_train_data(sample_k=len(benchmark._train_data), seed=eseed)
        for i in range(0, len(train), batch_size):
            batch = train[i:i + batch_size]
            yield [collate(x) for x in batch], benchmark.get_labels(batch)
        epoch += 1


def _wrap(text: str, indent: str) -> str:
    return "\n".join(textwrap.fill(text, width=96, initial_indent=indent,
                                   subsequent_indent=indent) for _ in [0])


def print_diff(before: dict, after: dict) -> None:
    changed = [n for n in before if before[n] != after[n]]
    if not changed:
        print("  (no instruction changed)")
        return
    for name in changed:
        print(f"\n  ┌─ agent `{name}` — proposed rewrite")
        print("  │ BEFORE:")
        print(_wrap(before[name], "  │   "))
        print("  │ AFTER:")
        print(_wrap(after[name], "  │   "))
        print("  └─")
    unchanged = [n for n in before if n not in changed]
    if unchanged:
        print(f"\n  (unchanged: {', '.join(unchanged)})")


def print_scores(accepted: dict, proposed: dict, shown=None, worse_when_up=None) -> None:
    shown = DEFAULT_SHOWN if shown is None else shown
    worse_when_up = DEFAULT_WORSE_WHEN_UP if worse_when_up is None else worse_when_up
    print("\n  score on dev  (accepted → proposed):")
    for k in shown:
        o, n = accepted.get(k, 0.0), proposed.get(k, 0.0)
        d = n - o
        if abs(d) < 1e-9:
            tag = "="
        elif (d > 0) != (k in worse_when_up):
            tag = "better"
        else:
            tag = "WORSE"
        flag = "   <- higher is worse" if k in worse_when_up else ""
        print(f"    {k:<15} {o:6.3f} → {n:6.3f}   ({tag}){flag}")


def instruction_violations(graph: WorkFlowGraph, max_words: int = 150) -> list[str]:
    """Enforce the limit that EvoAgentX only passes to the optimizer as prose."""
    violations = []
    for name, instruction in instructions(graph).items():
        count = len(instruction.split())
        if count > max_words:
            violations.append(f"{name}: {count} words (maximum {max_words})")
    return violations


def print_violations(violations: list[str]) -> None:
    if not violations:
        print("\n  structural checks: PASS")
        return
    print("\n  structural checks: FAIL")
    for violation in violations:
        print(f"    - {violation}")
    print("  Normal acceptance is blocked; an explicit constraint override is required.")


def ask() -> str:
    while True:
        try:
            r = input("\n  commit this evolution?  [a]ccept  [r]eject  [q]uit: ").strip().lower()
        except EOFError:
            return "q"
        if r in ("a", "accept", "y", "yes"):
            return "a"
        if r in ("r", "reject", "n", "no"):
            return "r"
        if r in ("q", "quit", "stop"):
            return "q"
        print("  please type a, r, or q.")


PENDING_DIR = Path("./optimized/_pending")


def build(args):
    """Construct the task, benchmark, optimizer and batch iterator. No API calls yet."""
    task = load_task(args.task, args.job)
    benchmark = task.make_benchmark(judge=task.judge_llm, smoothing=args.smoothing)

    # Start from the current accepted graph if one exists, so proposals chain.
    current = Path(f"./optimized/{benchmark.name}_evolved_current.json")
    if current.exists():
        graph = WorkFlowGraph.from_dict(json.loads(current.read_text()))
        print(f"(continuing from your accepted graph: {current})")
    else:
        graph = task.build_graph()

    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(graph, llm_config=task.executor_config)
    evaluator = RawOutputEvaluator(
        llm=task.executor_llm, agent_manager=agent_manager,
        collate_func=task.collate, output_postprocess_func=lambda o: o,
        num_workers=4, verbose=False,
    )
    optimizer = TextGradOptimizer(
        graph=graph, optimize_mode="all",
        executor_llm=task.executor_llm, optimizer_llm=task.optimizer_llm,
        batch_size=args.batch_size, max_steps=max(args.max_steps, 1),
        evaluator=evaluator, eval_every_n_steps=None, eval_rounds=1,
        rollback=False,               # the human is the rollback, not the mean score
        save_path="./optimized", save_interval=None,
        constraints=["The system prompt must not exceed 150 words."],
    )
    optimizer._init_textgrad(benchmark, use_answers=True)
    batches = data_iterator(benchmark, task.collate, args.batch_size, seed=args.seed)
    return task, benchmark, optimizer, batches


def _summary(m, shown=None):
    shown = DEFAULT_SHOWN if shown is None else shown
    return "  ".join(f"{k}={m.get(k,0):.3f}" for k in shown)


def require_no_pending() -> None:
    """Refuse to replace a proposal that has not received a human decision."""
    meta_path = PENDING_DIR / "meta.json"
    pending_files = list(PENDING_DIR.glob("*.json")) if PENDING_DIR.exists() else []
    if not pending_files:
        return
    if not meta_path.exists():
        raise SystemExit(
            "Incomplete pending proposal files exist in optimized/_pending. "
            "Inspect them before removing or recovering them; they will not be overwritten."
        )
    try:
        meta = json.loads(meta_path.read_text())
        label = f" for {meta.get('name', 'an unknown task')}"
    except (OSError, json.JSONDecodeError):
        label = ""
    raise SystemExit(
        "A proposal%s is already pending. Decide it before generating another:\n"
        "  python evolve.py --decide accept\n"
        "  python evolve.py --decide reject" % label
    )


def next_generation(name: str) -> int:
    """Return a never-before-used accepted generation number."""
    generations = []
    for path in Path("./optimized").glob(f"{name}_gen*_accepted.json"):
        try:
            generations.append(int(path.stem.rsplit("_gen", 1)[1].split("_", 1)[0]))
        except (IndexError, ValueError):
            pass
    return max(generations, default=0) + 1


def run_propose_only(args):
    """Generate ONE real evolution, show it, persist it as pending, and stop.
    Nothing is committed - `--decide` does that after you've looked."""
    require_no_pending()
    task, benchmark, optimizer, batches = build(args)
    shown, worse_when_up = metric_config(task)
    max_words = int(getattr(task, "max_instruction_words", 150))
    base_cfg = deepcopy(optimizer.graph.get_config())
    before = instructions(optimizer.graph)

    print(f"\ntask: {args.task}   |   dev examples: {len(benchmark.get_dev_data())}   |   smoothing: {args.smoothing}")
    print("scoring the CURRENT graph on dev ...")
    base_metrics = optimizer.evaluate(dataset=benchmark, eval_mode="dev")
    print("  current:", _summary(base_metrics, shown))

    print("\n### Generating one evolution (1 optimizer call) ...")
    inputs, labels = next(batches)
    optimizer.step(inputs, labels, benchmark, use_answers=True)
    after = instructions(optimizer.graph)
    if before == after:
        print("  optimizer returned no change on this batch. Nothing to decide; try again.")
        return
    print("  scoring the proposal on dev ...")
    proposed_metrics = optimizer.evaluate(dataset=benchmark, eval_mode="dev")
    cand_cfg = deepcopy(optimizer.graph.get_config())
    violations = instruction_violations(optimizer.graph, max_words)

    print(f"\n{'─' * 98}\nPROPOSED EVOLUTION")
    print_diff(before, after)
    print_scores(base_metrics, proposed_metrics, shown, worse_when_up)
    print_violations(violations)

    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    (PENDING_DIR / "base.json").write_text(json.dumps(base_cfg))
    (PENDING_DIR / "candidate.json").write_text(json.dumps(cand_cfg))
    (PENDING_DIR / "meta.json").write_text(json.dumps({
        "task": args.task, "name": benchmark.name,
        "base_metrics": base_metrics, "proposed_metrics": proposed_metrics,
        "changed_agents": [n for n in before if before[n] != after[n]],
        "shown_metrics": shown,
        "worse_when_up": sorted(worse_when_up),
        "max_instruction_words": max_words,
        "constraint_violations": violations,
    }, indent=2))
    print(f"\n{'=' * 98}")
    print("This proposal is PENDING. Nothing has changed in your runtime graph.")
    print(f"  accept it:  python evolve.py --decide accept   (--task {args.task})")
    print(f"  reject it:  python evolve.py --decide reject   (--task {args.task})")


def run_decide(args):
    """Apply the pending proposal's fate: accept promotes it; reject discards it."""
    if not (PENDING_DIR / "meta.json").exists():
        print("No pending proposal. Run `python evolve.py --propose-only` first.")
        return
    meta = json.loads((PENDING_DIR / "meta.json").read_text())
    name = meta["name"]
    current = Path(f"./optimized/{name}_evolved_current.json")
    shown = meta.get("shown_metrics", DEFAULT_SHOWN)

    if args.decide == "accept":
        cand = (PENDING_DIR / "candidate.json").read_text()
        candidate_graph = WorkFlowGraph.from_dict(json.loads(cand))
        violations = instruction_violations(
            candidate_graph, int(meta.get("max_instruction_words", 150))
        )
        if violations and not args.override_constraints:
            print("Acceptance refused because the candidate violates hard constraints:")
            for violation in violations:
                print(f"  - {violation}")
            print("The proposal remains pending. Review it, then either reject it or explicitly override:")
            print("  python evolve.py --decide reject")
            print("  python evolve.py --decide accept --override-constraints")
            return
        current.write_text(cand)
        # also snapshot as an immutable generation
        gen = next_generation(name)
        Path(f"./optimized/{name}_gen{gen}_accepted.json").write_text(cand)
        print(f"✓ accepted. changed agents: {', '.join(meta['changed_agents'])}")
        print(f"  dev score:  {_summary(meta['base_metrics'], shown)}")
        print(f"          ->  {_summary(meta['proposed_metrics'], shown)}")
        print(f"  accepted graph pointer -> {current}")
        print(f"  immutable snapshot     -> ./optimized/{name}_gen{gen}_accepted.json")
    else:
        print("✗ rejected. The proposal is discarded; your runtime graph is unchanged.")
        if current.exists():
            print(f"  accepted graph still -> {current}")

    for f in PENDING_DIR.glob("*.json"):
        f.unlink()
    print("  (pending cleared)")


def run_interactive(args):
    """Original loop: propose -> ask a/r/q at the terminal -> repeat."""
    require_no_pending()
    task, benchmark, optimizer, batches = build(args)
    shown, worse_when_up = metric_config(task)
    max_words = int(getattr(task, "max_instruction_words", 150))
    accepted_metrics = optimizer.evaluate(dataset=benchmark, eval_mode="dev")
    accepted_cfg = deepcopy(optimizer.graph.get_config())
    print(f"\ntask: {args.task}   |   dev examples: {len(benchmark.get_dev_data())}   |   smoothing: {args.smoothing}")
    print("  baseline:", _summary(accepted_metrics, shown))
    print("\n" + "=" * 98)
    print("Proposing one evolution at a time. Nothing is kept unless you accept it.")
    print("=" * 98)

    accepted = 0
    for i in range(1, args.max_steps + 1):
        before = instructions(optimizer.graph)
        inputs, labels = next(batches)
        print(f"\n\n### Evolution {i}: generating a rewrite (1 optimizer call) ...")
        optimizer.step(inputs, labels, benchmark, use_answers=True)
        after = instructions(optimizer.graph)
        if before == after:
            print("  optimizer returned no change on this batch; moving on.")
            continue
        proposed_metrics = optimizer.evaluate(dataset=benchmark, eval_mode="dev")
        violations = instruction_violations(optimizer.graph, max_words)
        print(f"\n{'─' * 98}\nPROPOSED EVOLUTION {i}")
        print_diff(before, after)
        print_scores(accepted_metrics, proposed_metrics, shown, worse_when_up)
        print_violations(violations)

        choice = ask()
        if choice == "a":
            if violations:
                optimizer.graph = WorkFlowGraph.from_dict(accepted_cfg)
                optimizer._create_textgrad_agents()
                print("  ✗ acceptance blocked by structural constraints; proposal reverted.")
                print("    Use --propose-only and --override-constraints for an explicit override.")
                continue
            accepted += 1
            accepted_cfg = deepcopy(optimizer.graph.get_config())
            accepted_metrics = proposed_metrics
            generation = next_generation(benchmark.name)
            path = f"./optimized/{benchmark.name}_gen{generation}_accepted.json"
            optimizer.save(path)
            print(f"  ✓ accepted. saved this version -> {path}")
        elif choice == "r":
            optimizer.graph = WorkFlowGraph.from_dict(accepted_cfg)
            optimizer._create_textgrad_agents()
            print("  ✗ rejected. reverted to your last accepted version.")
        else:
            optimizer.graph = WorkFlowGraph.from_dict(accepted_cfg)
            optimizer._create_textgrad_agents()
            print("  quitting at your request.")
            break

    optimizer.graph = WorkFlowGraph.from_dict(accepted_cfg)
    final = f"./optimized/{benchmark.name}_evolved_current.json"
    optimizer.save(final)
    print(f"\n{'=' * 98}\ndone. {accepted} evolution(s) accepted.")
    print(f"current accepted graph -> {final}")
    print("  final dev score:", _summary(accepted_metrics, shown))


def main():
    ap = argparse.ArgumentParser(description="Human-gated prompt evolution.")
    ap.add_argument("--task", default="ticket_task", help="task module (default: ticket_task)")
    ap.add_argument("--job", help="path to a reusable job.json (uses generic_task)")
    ap.add_argument("--max-steps", type=int, default=8, help="max evolutions to propose (interactive)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--smoothing", type=float, default=1.0, help="Beta smoothing; 0 once you have 50+ examples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--propose-only", action="store_true",
                    help="generate ONE evolution, show it, save as pending, and stop")
    ap.add_argument("--decide", choices=["accept", "reject"],
                    help="apply the fate of the pending proposal from --propose-only")
    ap.add_argument("--override-constraints", action="store_true",
                    help="with --decide accept, explicitly accept a structurally invalid prompt")
    args = ap.parse_args()

    if args.job:
        if args.task != "ticket_task":
            ap.error("Use either --task or --job, not both.")
        args.task = "generic_task"
    if args.override_constraints and args.decide != "accept":
        ap.error("--override-constraints is valid only with --decide accept.")

    if args.decide:
        run_decide(args)
    elif args.propose_only:
        run_propose_only(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
