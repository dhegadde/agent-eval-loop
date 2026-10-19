"""The default task for evolve.py: support-ticket triage + reply.

This is the TASK PLUGIN CONTRACT. To point evolve.py at a different job, write a
module exposing exactly these names and run `python evolve.py --task your_module`:

    build_graph()                       -> a SequentialWorkFlowGraph whose task
                                           prompt_templates carry the instructions
                                           TextGrad will rewrite.
    make_benchmark(judge, smoothing)    -> a Benchmark with evaluate(pred, label)
                                           returning a NUMERIC-ONLY metrics dict
                                           (route any text to a side file; GOTCHAS #8).
    collate(example)                    -> {graph_input_name: value} for one row.
    executor_config, executor_llm,      -> the three LLM roles (cheap executor,
    optimizer_llm, judge_llm               strong optimizer, separate judge).

Nothing here is ticket-specific to evolve.py itself - swap this file and the whole
human-gated loop runs on your task unchanged. See docs/EVOLVE.md.
"""

from benchmark_tickets import TicketBenchmark
from setup_swarm import (  # re-exported for evolve.py's task contract
    build_graph,
    executor_config,
    executor_llm,
    judge_llm,
    optimizer_llm,
)

__all__ = [
    "build_graph", "make_benchmark", "collate",
    "executor_config", "executor_llm", "optimizer_llm", "judge_llm",
]


def make_benchmark(judge=None, smoothing: float = 1.0) -> TicketBenchmark:
    return TicketBenchmark(
        path="data/tickets.jsonl",
        judge_llm=judge if judge is not None else judge_llm,
        smoothing=smoothing,
    )


def collate(example: dict) -> dict:
    """Benchmark row -> workflow inputs. Keys must match the graph's declared inputs."""
    return {"ticket": example["ticket"]}
