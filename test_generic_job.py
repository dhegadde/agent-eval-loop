"""Zero-API tests for the reusable job layer."""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "dummy")

import generic_task
from generic_benchmark import GenericTextBenchmark
from job_specs import dataset_report, fingerprint_test_set, load_examples, load_job_spec


def fixture(root: Path) -> Path:
    spec = {
        "schema_version": 1,
        "name": "normalize_answer",
        "goal": "Normalize a short answer.",
        "instruction": "Return the concise normalized answer without adding facts.",
        "input": {"name": "input_text", "description": "Raw answer."},
        "output": {"name": "output_text", "description": "Normalized answer."},
        "evaluation": {"mode": "exact", "case_sensitive": False},
        "data": "examples.jsonl",
    }
    path = root / "job.json"
    path.write_text(json.dumps(spec))
    rows = [
        {"id": "tr1", "input": " YES ", "reference": "yes", "split": "train"},
        {"id": "tr2", "input": " NO ", "reference": "no", "split": "train"},
        {"id": "dv1", "input": " OK ", "reference": "ok", "split": "dev"},
        {"id": "dv2", "input": " MAYBE ", "reference": "maybe", "split": "dev"},
        {"id": "te1", "input": " TRUE ", "reference": "true", "split": "test"},
    ]
    (root / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = fixture(Path(tmp))
        spec_path, spec = load_job_spec(path)
        rows = load_examples(spec_path, spec)
        report = dataset_report(rows)
        assert report["ready_to_run"] is True
        assert report["measurement_ready"] is False
        assert len(fingerprint_test_set(rows)) == 64

        benchmark = GenericTextBenchmark(spec_path, spec, judge_llm=None, smoothing=0)
        label = benchmark.get_label(benchmark.get_dev_data()[0])
        good = benchmark.evaluate({"output_text": " OK "}, label)
        bad = benchmark.evaluate({"output_text": "wrong"}, label)
        assert good["raw_score"] == 1.0
        assert bad["raw_score"] == 0.0
        assert all(isinstance(value, float) for value in good.values())

        generic_task.configure(str(path))
        graph = generic_task.build_graph()
        assert [node.name for node in graph.nodes] == ["perform_job"]
        assert generic_task.collate(rows[0]) == {"input_text": " YES "}
        assert generic_task.make_benchmark(judge=None).name == "normalize_answer"

    print("ALL GENERIC-JOB TESTS PASSED (zero API calls).")


if __name__ == "__main__":
    main()

