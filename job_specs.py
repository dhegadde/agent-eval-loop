"""Validation and loading for reusable personal-job definitions.

A job is intentionally data, not Python code: one ``job.json`` plus a JSONL file
of real examples.  ``generic_task.py`` compiles that definition into the small
task-module contract consumed by ``evolve.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
FIELD_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
VALID_SPLITS = {"train", "dev", "test"}
VALID_EVALUATION_MODES = {"judge", "exact"}


class JobSpecError(ValueError):
    """Raised when a job definition or dataset would make evaluation unsafe."""


def load_job_spec(path: str | Path) -> tuple[Path, dict[str, Any]]:
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise JobSpecError(f"Job definition does not exist: {spec_path}")
    try:
        spec = json.loads(spec_path.read_text())
    except json.JSONDecodeError as exc:
        raise JobSpecError(f"Invalid JSON in {spec_path}: {exc}") from exc
    validate_job_spec(spec)
    return spec_path, spec


def validate_job_spec(spec: Any) -> None:
    if not isinstance(spec, dict):
        raise JobSpecError("job.json must contain one JSON object.")

    required_strings = ("name", "goal", "instruction")
    for key in required_strings:
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise JobSpecError(f"job.json field {key!r} must be a non-empty string.")

    if not NAME_RE.fullmatch(spec["name"]):
        raise JobSpecError(
            "job name must be 2-64 lowercase letters, numbers, or underscores "
            "and start with a letter."
        )

    for side in ("input", "output"):
        field = spec.get(side)
        if not isinstance(field, dict):
            raise JobSpecError(f"job.json field {side!r} must be an object.")
        for key in ("name", "description"):
            if not isinstance(field.get(key), str) or not field[key].strip():
                raise JobSpecError(f"{side}.{key} must be a non-empty string.")
        if not FIELD_RE.fullmatch(field["name"]):
            raise JobSpecError(f"{side}.name must be a lowercase identifier.")

    if spec["input"]["name"] == spec["output"]["name"]:
        raise JobSpecError("input.name and output.name must differ (EvoAgentX GOTCHAS #6).")

    data_path = spec.get("data", "examples.jsonl")
    if not isinstance(data_path, str) or not data_path.strip():
        raise JobSpecError("data must be a non-empty path string.")
    if Path(data_path).is_absolute():
        raise JobSpecError("data must be relative to job.json so the job stays portable.")

    max_words = spec.get("max_instruction_words", 150)
    if not isinstance(max_words, int) or not 20 <= max_words <= 2000:
        raise JobSpecError("max_instruction_words must be an integer from 20 to 2000.")
    instruction_words = len(spec["instruction"].split())
    if instruction_words > max_words:
        raise JobSpecError(
            f"seed instruction has {instruction_words} words; maximum is {max_words}."
        )

    evaluation = spec.get("evaluation")
    if not isinstance(evaluation, dict):
        raise JobSpecError("evaluation must be an object.")
    mode = evaluation.get("mode")
    if mode not in VALID_EVALUATION_MODES:
        raise JobSpecError("evaluation.mode must be 'judge' or 'exact'.")
    if mode == "judge":
        rubric = evaluation.get("rubric")
        if not isinstance(rubric, str) or not rubric.strip():
            raise JobSpecError("evaluation.rubric is required for judge mode.")
        hard_rules = evaluation.get("hard_rules", [])
        if not isinstance(hard_rules, list) or not all(
            isinstance(rule, str) and rule.strip() for rule in hard_rules
        ):
            raise JobSpecError("evaluation.hard_rules must be a list of non-empty strings.")


def data_path_for(spec_path: Path, spec: dict[str, Any]) -> Path:
    return (spec_path.parent / spec.get("data", "examples.jsonl")).resolve()


def load_examples(spec_path: Path, spec: dict[str, Any]) -> list[dict[str, str]]:
    path = data_path_for(spec_path, spec)
    if not path.is_file():
        raise JobSpecError(f"Examples file does not exist: {path}")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JobSpecError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise JobSpecError(f"{path}:{lineno}: each line must be a JSON object.")
        for field in ("id", "input", "reference", "split"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise JobSpecError(f"{path}:{lineno}: {field!r} must be a non-empty string.")
        if row["split"] not in VALID_SPLITS:
            raise JobSpecError(
                f"{path}:{lineno}: split must be one of {sorted(VALID_SPLITS)}."
            )
        if row["id"] in seen:
            raise JobSpecError(f"{path}:{lineno}: duplicate id {row['id']!r}.")
        seen.add(row["id"])
        rows.append({key: str(value) for key, value in row.items()})
    return rows


def dataset_report(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(row["split"] for row in rows)
    return {
        "total": len(rows),
        "train": counts["train"],
        "dev": counts["dev"],
        "test": counts["test"],
        "ready_to_run": counts["train"] >= 2 and counts["dev"] >= 2,
        "measurement_ready": len(rows) >= 50 and all(counts[s] >= 10 for s in VALID_SPLITS),
    }


def fingerprint_test_set(rows: list[dict[str, str]]) -> str:
    """Stable audit fingerprint; changing held-out rows changes this value."""
    held_out = [row for row in rows if row["split"] == "test"]
    canonical = json.dumps(held_out, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
