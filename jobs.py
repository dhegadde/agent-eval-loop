"""Create, fill, and validate portable jobs for the human-gated optimizer.

This utility never manufactures examples. References must be written or approved
by a human, because synthetic targets turn the optimizer into a style imitator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from job_specs import (
    JobSpecError,
    data_path_for,
    dataset_report,
    fingerprint_test_set,
    load_examples,
    load_job_spec,
    validate_job_spec,
)


def _value(text: str | None, path: str | None, label: str) -> str:
    if text is not None and path is not None:
        raise JobSpecError(f"Use --{label} or --{label}-file, not both.")
    if path is not None:
        value = Path(path).expanduser().read_text().strip()
    elif text is not None:
        value = text.strip()
    else:
        raise JobSpecError(f"One of --{label} or --{label}-file is required.")
    if not value:
        raise JobSpecError(f"{label} cannot be empty.")
    return value


def cmd_init(args) -> None:
    evaluation = {"mode": args.evaluation}
    if args.evaluation == "judge":
        if not args.rubric:
            raise JobSpecError("--rubric is required for judge evaluation.")
        evaluation.update({
            "rubric": args.rubric,
            "hard_rules": args.hard_rule or [],
        })
    else:
        evaluation["case_sensitive"] = bool(args.case_sensitive)

    spec = {
        "schema_version": 1,
        "name": args.name,
        "goal": args.goal,
        "instruction": args.instruction,
        "input": {"name": "input_text", "description": args.input_description},
        "output": {"name": "output_text", "description": args.output_description},
        "evaluation": evaluation,
        "data": "examples.jsonl",
        "max_instruction_words": 150,
    }
    validate_job_spec(spec)
    root = Path(args.root).expanduser().resolve() / args.name
    if root.exists():
        raise JobSpecError(f"Refusing to overwrite existing job directory: {root}")
    root.mkdir(parents=True)
    (root / "job.json").write_text(json.dumps(spec, indent=2) + "\n")
    (root / "examples.jsonl").write_text("")
    print(f"Created {root / 'job.json'}")
    print("No examples were generated. Add real, human-approved examples next:")
    print(f"  python jobs.py add {root / 'job.json'} --split train --input ... --reference ...")


def cmd_add(args) -> None:
    spec_path, spec = load_job_spec(args.job)
    rows = load_examples(spec_path, spec)
    test_lock = spec_path.parent / "test.lock.json"
    if args.split == "test" and test_lock.exists():
        raise JobSpecError(
            f"The test set is frozen by {test_lock}; refusing to add a test example."
        )

    source = _value(args.input, args.input_file, "input")
    reference = _value(args.reference, args.reference_file, "reference")
    example_id = args.id or f"ex-{len(rows) + 1:04d}"
    if any(row["id"] == example_id for row in rows):
        raise JobSpecError(f"Example id already exists: {example_id}")

    row = {"id": example_id, "input": source, "reference": reference, "split": args.split}
    path = data_path_for(spec_path, spec)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Added {example_id} to {args.split}: {path}")


def cmd_status(args) -> None:
    spec_path, spec = load_job_spec(args.job)
    rows = load_examples(spec_path, spec)
    report = dataset_report(rows)
    print(f"job: {spec['name']} — {spec['goal']}")
    print(
        f"examples: {report['total']} total | {report['train']} train | "
        f"{report['dev']} dev | {report['test']} test"
    )
    if report["measurement_ready"]:
        print("measurement: READY (50+ total and at least 10 in every split)")
    elif report["ready_to_run"]:
        print("measurement: runnable but noisy; keep collecting toward 50+")
    else:
        print("measurement: not runnable; needs at least 2 train and 2 dev examples")

    lock_path = spec_path.parent / "test.lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        current = fingerprint_test_set(rows)
        state = "unchanged" if lock.get("sha256") == current else "CHANGED"
        print(f"test set: frozen, {state} ({lock_path})")
        if state == "CHANGED":
            raise JobSpecError("Held-out test data changed after it was frozen.")
    else:
        print("test set: not frozen; run `python jobs.py freeze-test <job.json>` when curated")


def cmd_freeze_test(args) -> None:
    spec_path, spec = load_job_spec(args.job)
    rows = load_examples(spec_path, spec)
    if not any(row["split"] == "test" for row in rows):
        raise JobSpecError("Cannot freeze an empty test set.")
    lock_path = spec_path.parent / "test.lock.json"
    if lock_path.exists():
        raise JobSpecError(f"Test set is already frozen: {lock_path}")
    payload = {
        "sha256": fingerprint_test_set(rows),
        "test_ids": [row["id"] for row in rows if row["split"] == "test"],
        "note": "Do not edit held-out test rows while optimizing prompts.",
    }
    lock_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Frozen {len(payload['test_ids'])} held-out examples -> {lock_path}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Manage real datasets for reusable text jobs.")
    sub = ap.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create a new job definition (no synthetic examples)")
    init.add_argument("name", help="lowercase identifier, e.g. job_application_tailoring")
    init.add_argument("--root", default="jobs")
    init.add_argument("--goal", required=True)
    init.add_argument("--instruction", required=True, help="seed prompt that TextGrad may rewrite")
    init.add_argument("--input-description", required=True)
    init.add_argument("--output-description", required=True)
    init.add_argument("--evaluation", choices=["judge", "exact"], default="judge")
    init.add_argument("--rubric", help="quality rubric for judge mode")
    init.add_argument("--hard-rule", action="append", help="explicit hard failure; repeatable")
    init.add_argument("--case-sensitive", action="store_true", help="exact mode only")
    init.set_defaults(func=cmd_init)

    add = sub.add_parser("add", help="append one real, human-approved example")
    add.add_argument("job", help="path to job.json")
    add.add_argument("--id")
    add.add_argument("--split", choices=["train", "dev", "test"], required=True)
    add.add_argument("--input")
    add.add_argument("--input-file")
    add.add_argument("--reference")
    add.add_argument("--reference-file")
    add.set_defaults(func=cmd_add)

    status = sub.add_parser("status", help="validate a job and report data readiness")
    status.add_argument("job", help="path to job.json")
    status.set_defaults(func=cmd_status)

    freeze = sub.add_parser("freeze-test", help="fingerprint the held-out test set")
    freeze.add_argument("job", help="path to job.json")
    freeze.set_defaults(func=cmd_freeze_test)
    return ap


def main() -> int:
    try:
        args = parser().parse_args()
        args.func(args)
        return 0
    except (JobSpecError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
