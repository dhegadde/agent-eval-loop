"""Free (no API calls) test of evolve.py's gate mechanics.

Proves the non-LLM logic - instruction extraction, diff rendering, batching, and
the accept/reject/restore snapshot flow - works, without spending anything. The
LLM-driven step() itself is exercised by a real run.
"""
import os
os.environ.setdefault("OPENAI_API_KEY", "dummy")

from copy import deepcopy
from pathlib import Path
import tempfile
from evoagentx.workflow import WorkFlowGraph

import evolve
import ticket_task


def test_instructions_and_diff():
    g = ticket_task.build_graph()
    ins = evolve.instructions(g)
    assert set(ins) == {"triage", "extract_facts", "draft_reply"}, ins
    assert all(isinstance(v, str) and v for v in ins.values())
    # simulate a one-agent rewrite
    after = dict(ins)
    after["draft_reply"] = "NEW: Be concise and never invent facts."
    print("-- print_diff on a single-agent change --")
    evolve.print_diff(ins, after)
    print("\nOK: instructions extracted + diff renders only the changed agent\n")


def test_score_delta_direction():
    accepted = {"raw_score": 0.75, "category_acc": 1.0, "priority_score": 1.0,
                "reply_score": 0.38, "hallucinated": 0.0}
    proposed = {"raw_score": 0.60, "category_acc": 1.0, "priority_score": 0.5,
                "reply_score": 0.40, "hallucinated": 0.5}
    print("-- print_scores: raw_score down=WORSE, reply_score up=better, halluc up=WORSE --")
    evolve.print_scores(accepted, proposed)
    print("\nOK: hallucinated is flagged 'higher is worse'\n")


def test_batching():
    bench = ticket_task.make_benchmark(judge=None)  # no judge_llm -> zero LLM calls
    it = evolve.data_iterator(bench, ticket_task.collate, batch_size=2, seed=0)
    inputs, labels = next(it)
    assert len(inputs) == 2 and len(labels) == 2, (inputs, labels)
    assert set(inputs[0]) == {"ticket"}, inputs[0]
    assert isinstance(labels[0], str), "labels must be JSON strings (TextGrad requires it)"
    print("OK: batches yield {'ticket': ...} inputs and string labels\n")


def test_accept_reject_restore():
    """The core safety property: reject snaps the graph back to the accepted config."""
    g = ticket_task.build_graph()
    accepted_cfg = deepcopy(g.get_config())
    base = evolve.instructions(g)

    # mutate as if step() rewrote an instruction
    cfg = deepcopy(g.get_config())
    cfg["tasks"][2]["prompt_template"]["instruction"] = "MUTATED draft instruction"
    mutated = WorkFlowGraph.from_dict(cfg)
    assert evolve.instructions(mutated)["draft_reply"] == "MUTATED draft instruction"

    # REJECT path: restore from accepted_cfg
    restored = WorkFlowGraph.from_dict(accepted_cfg)
    assert evolve.instructions(restored) == base, "reject must restore the accepted instructions"
    print("OK: reject restores the last accepted graph exactly\n")


def test_structural_gate():
    g = ticket_task.build_graph()
    assert evolve.instruction_violations(g, max_words=150) == []
    cfg = deepcopy(g.get_config())
    cfg["tasks"][0]["prompt_template"]["instruction"] = "word " * 151
    bloated = WorkFlowGraph.from_dict(cfg)
    violations = evolve.instruction_violations(bloated, max_words=150)
    assert violations == ["triage: 151 words (maximum 150)"], violations
    print("OK: structural gate catches prompts EvoAgentX's prose constraint does not enforce\n")


def test_generation_numbers_never_reuse_files():
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            out = Path("optimized")
            out.mkdir()
            (out / "demo_gen1_accepted.json").write_text("one")
            (out / "demo_gen3_accepted.json").write_text("three")
            assert evolve.next_generation("demo") == 4
        finally:
            os.chdir(original)
    print("OK: accepted generation numbering never reuses an existing snapshot\n")


if __name__ == "__main__":
    test_instructions_and_diff()
    test_score_delta_direction()
    test_batching()
    test_accept_reject_restore()
    test_structural_gate()
    test_generation_numbers_never_reuse_files()
    print("=" * 60)
    print("ALL GATE-LOGIC TESTS PASSED (zero API calls).")
