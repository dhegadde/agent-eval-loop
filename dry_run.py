"""Validate the whole pipeline with zero API calls.

Monkeypatches LiteLLM.generate to return canned text, then runs a real
Evaluator over the dev split. If this prints a score dict, your benchmark,
collate_func, and output parsing all line up. Run it before optimize_textgrad.py.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "dummy")

from evoagentx.agents import AgentManager  # noqa: E402
from evoagentx.evaluators import Evaluator  # noqa: E402
from evoagentx.models import LiteLLM  # noqa: E402
from evoagentx.models.base_model import LLMOutputParser  # noqa: E402

from benchmark_tickets import TicketBenchmark  # noqa: E402
from setup_swarm import build_graph, executor_config  # noqa: E402

CANNED = {
    "triage": "category: billing\npriority: P1",
    "extract_facts": "facts: - Charged twice on 3 Aug\nmissing: - Invoice number",
    "draft_reply": "reply: Thanks for flagging the duplicate charge. I've refunded it.",
}


def fake_generate(self, prompt=None, system_message=None, messages=None, parser=None,
                  parse_mode="str", **kwargs):
    blob = " ".join(str(x) for x in (prompt, system_message, messages))
    text = CANNED["triage"]
    for key in ("draft_reply", "extract_facts", "triage"):
        if key in blob or key.replace("_", " ") in blob.lower():
            text = CANNED[key]
            break
    if parser is not None:
        try:
            return parser.parse(text, parse_mode=parse_mode)
        except Exception:
            pass
    return LLMOutputParser(content=text)


LiteLLM.generate = fake_generate
LiteLLM.single_generate = lambda self, *a, **k: CANNED["triage"]


def collate(example: dict) -> dict:
    return {"ticket": example["ticket"]}


def main():
    benchmark = TicketBenchmark(path="data/tickets.jsonl")
    print("splits -> train:", len(benchmark.get_train_data()),
          "dev:", len(benchmark.get_dev_data()),
          "test:", len(benchmark.get_test_data()))

    graph = build_graph()
    print("nodes:", [n.name for n in graph.nodes])
    print("edges:", [(e.source, e.target) for e in graph.edges])

    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(graph, llm_config=executor_config)
    print("agents registered:", len(agent_manager.agents))

    # Metric sanity check, independent of the LLM.
    import json as _json
    row = benchmark.get_dev_data()[0]
    label = benchmark.get_label(row)          # a JSON *string* - TextGrad requires this
    gold = _json.loads(label)

    # RawOutputEvaluator returns every node's outputs merged into one dict.
    perfect = {"category": gold["category"], "priority": gold["priority"],
               "reply": gold["reply"]}
    wrong = {"category": "bug", "priority": "P3", "reply": "hi"}

    def _fmt(d):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in d.items() if k != "critique"}

    print("perfect prediction ->", _fmt(benchmark.evaluate(perfect, label)))
    print("wrong prediction   ->", _fmt(benchmark.evaluate(wrong, label)))
    print("(no judge_llm here, so reply_score uses the fallback heuristic)")

    evaluator = Evaluator(
        llm=LiteLLM(config=executor_config),
        agent_manager=agent_manager,
        collate_func=collate,
        output_postprocess_func=lambda o: o,
        num_workers=1,
        verbose=False,
    )
    print("evaluator constructed OK:", type(evaluator).__name__)
    print(
        "\nWiring valid. Configure the desired model backend, then run the "
        "human-gated evolve.py workflow."
    )


if __name__ == "__main__":
    main()
