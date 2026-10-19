"""Compile a portable job.json into evolve.py's task-plugin contract."""

from __future__ import annotations

from pathlib import Path

from evoagentx.prompts.template import StringTemplate
from evoagentx.workflow import SequentialWorkFlowGraph

from generic_benchmark import GenericTextBenchmark
from job_specs import load_job_spec
from setup_swarm import executor_config, executor_llm, judge_llm, optimizer_llm


_spec_path: Path | None = None
_spec: dict | None = None

shown_metrics = ["raw_score", "quality_score", "hard_failure", "judge_error"]
worse_when_up = {"hard_failure", "judge_error"}
max_instruction_words = 150


def configure(path: str) -> None:
    global _spec_path, _spec, max_instruction_words
    _spec_path, _spec = load_job_spec(path)
    max_instruction_words = int(_spec.get("max_instruction_words", 150))


def _configured() -> tuple[Path, dict]:
    if _spec_path is None or _spec is None:
        raise RuntimeError("generic_task requires `python evolve.py --job path/to/job.json`.")
    return _spec_path, _spec


def build_graph() -> SequentialWorkFlowGraph:
    _, spec = _configured()
    task = {
        "name": "perform_job",
        "description": spec["goal"],
        "inputs": [{
            "name": spec["input"]["name"],
            "type": "string",
            "required": True,
            "description": spec["input"]["description"],
        }],
        "outputs": [{
            "name": spec["output"]["name"],
            "type": "string",
            "required": True,
            "description": spec["output"]["description"],
        }],
        "parse_mode": "json",
        "prompt_template": StringTemplate(instruction=spec["instruction"]),
    }
    return SequentialWorkFlowGraph(goal=spec["goal"], tasks=[task])


def make_benchmark(judge=None, smoothing: float = 1.0) -> GenericTextBenchmark:
    spec_path, spec = _configured()
    return GenericTextBenchmark(
        spec_path=spec_path,
        spec=spec,
        judge_llm=judge if judge is not None else judge_llm,
        smoothing=smoothing,
    )


def collate(example: dict) -> dict:
    _, spec = _configured()
    return {spec["input"]["name"]: example["input"]}


__all__ = [
    "configure", "build_graph", "make_benchmark", "collate",
    "executor_config", "executor_llm", "optimizer_llm", "judge_llm",
    "shown_metrics", "worse_when_up", "max_instruction_words",
]
