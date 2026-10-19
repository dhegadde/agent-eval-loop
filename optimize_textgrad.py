"""Offline optimization. Run on a schedule, never in the request path.

Now on OpenAI. Paid keys have far higher limits than Gemini's free tier, so the
throttle is relaxed and workers are parallel again. If you see 429s, lower
max_rpm - OpenAI usage tier 1 is 500 RPM, tier 2+ is much higher.
"""

# MUST be imported and installed before any LiteLLM object is constructed.
from rate_limit import install_rpm_limit

install_rpm_limit(max_rpm=200)  # safety net only; OpenAI tier 1 allows ~500 RPM

from evoagentx.agents import AgentManager  # noqa: E402
from evoagentx.optimizers.textgrad_optimizer import TextGradOptimizer  # noqa: E402

from benchmark_tickets import TicketBenchmark  # noqa: E402
from raw_evaluator import RawOutputEvaluator  # noqa: E402
from setup_swarm import (  # noqa: E402
    build_graph, executor_config, executor_llm, judge_llm, optimizer_llm,
)


def collate(example: dict) -> dict:
    """Benchmark row -> workflow inputs. Keys must match the graph's declared inputs."""
    return {"ticket": example["ticket"]}


def main():
    benchmark = TicketBenchmark(
        path="data/tickets.jsonl",
        judge_llm=judge_llm,   # replaces token-overlap scoring
        smoothing=1.0,         # set to 0 once you have 50+ examples
    )
    graph = build_graph()

    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(graph, llm_config=executor_config)

    # RawOutputEvaluator, not Evaluator: grades the end node's dict directly
    # instead of an LLM prose summary. Fixes scoring AND saves one call per example.
    evaluator = RawOutputEvaluator(
        llm=executor_llm,
        agent_manager=agent_manager,
        collate_func=collate,
        output_postprocess_func=lambda o: o,
        num_workers=4,          # lower to 1 if you start seeing 429s
        verbose=True,
    )

    optimizer = TextGradOptimizer(
        graph=graph,
        optimize_mode="all",
        executor_llm=executor_llm,
        optimizer_llm=optimizer_llm,
        batch_size=2,
        max_steps=8,            # 8 is reasonable once you have 50+ examples
        evaluator=evaluator,
        eval_every_n_steps=2,   # watch for drift as it optimizes
        eval_rounds=1,
        rollback=True,          # keep the best graph, discard regressions
        save_path="./optimized",
        save_interval=1,
        constraints=["The system prompt must not exceed 150 words."],
    )

    print("=== BASELINE (dev) ===")
    baseline = optimizer.evaluate(dataset=benchmark, eval_mode="dev")
    print(baseline)

    optimizer.optimize(dataset=benchmark, use_answers=True, seed=0)

    print("=== OPTIMIZED (test) ===")
    print(optimizer.evaluate(dataset=benchmark, eval_mode="test"))

    optimizer.save("./optimized/swarm_v2.json")
    print("saved -> ./optimized/swarm_v2.json")


if __name__ == "__main__":
    main()