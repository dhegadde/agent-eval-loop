# Reusable personal jobs

The framework can now compile a portable `job.json` into a one-agent EvoAgentX
workflow. This is the default shape for a new personal use case: one text input,
one text output, real examples, and either exact or judge-based scoring.

It does **not** mean arbitrary autonomy. A valid job is a repeated text task whose
output can be evaluated. Multi-input, structured-output, or multi-agent jobs can
still use a custom task module such as `ticket_task.py`.

## Pick a good first job

A useful first job has all four properties:

1. You do it repeatedly.
2. You can provide real inputs and human-approved outputs.
3. Better output compounds into saved time or better decisions.
4. You can name explicit hard failures.

Good personal candidates:

- tailor a cover letter or resume summary from a job description and a verified
  experience bank; hard rule: never fabricate experience;
- draft freelance proposals from a client brief and your real portfolio;
- turn interview questions into STAR-answer drafts grounded in your experience;
- rewrite recurring outreach in your own approved style.

Real-world outcomes such as callbacks, replies, or contracts are worth tracking,
but they are noisy and confounded. The current optimizer learns against reference
outputs and a rubric; it does **not** yet claim that a prompt caused a callback.

## Create a job

`jobs.py init` creates a definition and an empty dataset. It deliberately creates
no synthetic examples.

```bash
python jobs.py init job_application_tailoring \
  --goal "Tailor a truthful application draft to a supplied job description" \
  --instruction "Use only the supplied experience facts. Select the strongest relevant evidence and write a concise, specific application draft." \
  --input-description "A job description followed by the applicant's verified experience bank" \
  --output-description "A truthful tailored application draft" \
  --rubric "Reward relevance to the role, specificity, concise writing, and evidence from the experience bank." \
  --hard-rule "Do not claim any skill, achievement, employer, credential, or experience absent from the input."
```

That produces:

```text
jobs/job_application_tailoring/
├── job.json
└── examples.jsonl
```

For an exact-answer classification or normalization task, use
`--evaluation exact` and omit the rubric.

## Add real examples

Each reference must be written or approved by a human. Keep personally identifying
details redacted unless they are necessary and you are comfortable storing them.

```bash
python jobs.py add jobs/job_application_tailoring/job.json \
  --split train \
  --input-file /path/to/application-input.txt \
  --reference-file /path/to/approved-draft.txt
```

The JSONL schema is stable across all reusable jobs:

```json
{"id":"ex-0001","input":"...","reference":"...","split":"train"}
```

Use explicit splits so adding examples never reshuffles prior rows. At 50 examples,
a practical target is 30 train / 10 dev / 10 test. Include hard cases and failures,
not just examples the seed prompt already handles.

Check progress and freeze the test set once it is curated:

```bash
python jobs.py status jobs/job_application_tailoring/job.json
python jobs.py freeze-test jobs/job_application_tailoring/job.json
```

The freeze command stores a SHA-256 fingerprint and IDs in `test.lock.json`.
`jobs.py` then refuses to append new test examples, and `status` detects edits.

## Evolve it

Do not spend on evolution until the judge rubric has been checked on crafted pass
and fail cases. When ready:

```bash
python evolve.py --job jobs/job_application_tailoring/job.json --propose-only
python evolve.py --decide accept   # or reject
```

Only one proposal may be pending anywhere in the repo. A second proposal is
refused until the first receives your decision. Every candidate also receives a
real word-count check. EvoAgentX's own `constraints` field is merely prompt advice;
it did not enforce the limit in the completed autonomous run.

Ordinary acceptance is blocked when an instruction exceeds
`max_instruction_words` (150 by default). You retain final authority through a
deliberately explicit second action:

```bash
python evolve.py --decide accept --override-constraints
```

Use that rarely and only after reading the full diff.

