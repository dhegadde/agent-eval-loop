# EvoAgentX 0.1.4 — gotchas

Every item here cost a failed run to find. All were confirmed by reading
`site-packages/evoagentx/` source, not documentation. If you change code that
touches the framework, check this list first.

Verified against `evoagentx==0.1.4`, `dspy==3.3.1` (comparison only), Python 3.11.

---

## 0. The install is broken out of the box

`pip install evoagentx` under-declares its dependencies badly.
`evoagentx/tools/__init__.py` eagerly imports **every** integration it ships —
Docker, Selenium, MongoDB, Postgres, Telethon, five llama-index backends,
tkinter. So even `from evoagentx.models import LiteLLM` raises
`ModuleNotFoundError` until all of them are installed, despite using none.

Getting to a clean import took 25 extra pip packages plus one system package.
See `requirements-extra.txt`. Plus:

```bash
brew install python-tk@3.11     # macOS; evoagentx.hitl imports tkinter
sudo apt-get install python3-tk # Linux
```

Two traps with misleading errors:

- **`bson`** — if the standalone `bson` package is installed,
  `from bson import json_util` fails with `ImportError`, *not* a missing-module
  error. Uninstall `bson`, install `pymongo`, which ships the real one.
- **`fastmcp`** — may collide with a system PyJWT. Use
  `pip install fastmcp --ignore-installed PyJWT`.

The real gate is `python -c "from evoagentx.models import LiteLLM"`. If that
passes, ignore whatever else is on the list.

---

## 1. `openai_key`, not `api_key`

`LiteLLMConfig` infers the provider from the model string and requires the
**matching** key field. `api_key` exists but is for generic OpenAI-compatible
endpoints; passing your key there authenticates nothing and fails confusingly.

```python
LiteLLMConfig(model="gpt-5.4-nano", openai_key=KEY)              # OpenAI
LiteLLMConfig(model="gemini/gemini-3.6-flash", gemini_key=KEY)   # Gemini
```

Bare model names with no `/` prefix are treated as OpenAI. `litellm_model.py`
reads `gemini_key` and sets `os.environ["GEMINI_API_KEY"]` from it.

---

## 2. `parse_mode` defaults to `"str"` and destroys structure

`SequentialWorkFlowGraph` passes `parse_mode="str"` by default, which assigns the
**entire raw response** to **every** declared output field. Symptom: both
`category` and `priority` hold `"category: billing\npriority: P2"` and everything
scores 0.0.

Set `"parse_mode": "json"` on every task. The JSON parser handles code fences —
```` ```json {...} ``` ```` extracts cleanly (verified).

---

## 3. `prompt_template` silently overrides `prompt`

Supply both and `CustomizeAgent` logs a warning and **ignores `prompt` entirely**.
Any output-format rules written there never reach the model.

Don't supply `prompt` at all. With `parse_mode="json"` the template auto-generates
the schema instruction from each output's `description` field, which is rendered
verbatim into the prompt. That makes `description` load-bearing — it's where the
valid category and priority values actually come from.

---

## 4. TextGrad requires `prompt_template.instruction`

`TextGradAgent.__init__` reads `agent.actions[0].prompt_template.instruction`.
Supply only `prompt` and that attribute is `None`:

```
AttributeError: 'NoneType' object has no attribute 'instruction'
```

...raised *before* the first optimization step, after you've already paid for the
baseline eval. The `instruction` string is the only thing TextGrad rewrites, so
real task guidance belongs there.

---

## 5. `_get_label` must return a **string**

`TextGradOptimizer.step()` raises `ValueError: Label must be a string for
non-coding benchmarks` if it returns a dict. Serialize to JSON in `_get_label`,
deserialize in `evaluate()`. See `_as_label_dict()`.

---

## 6. A node cannot reuse its own input names as outputs

```
inputs and outputs share the following name(s), which is not allowed
```

Raised at graph construction. Blocks the obvious "pass the field through"
approach — which is fine, because you shouldn't do that anyway (#7).

---

## 7. The stock Evaluator grades an LLM summary, not your outputs

`Evaluator._execute_workflow_graph` calls
`workflow.execute(..., extract_output=True)`, which fires an **extra LLM call**
to summarize the run into prose. That prose is what reaches `benchmark.evaluate`.
The structured dict your agents produced is thrown away.

Two costs: fields score 0.0 despite the agents being right, and it's one extra
call per example per evaluation (~25% of request budget).

`extract_output=False` returns the raw dict for free. The stock Evaluator
hardcodes `True`, so `raw_evaluator.py` overrides the one method and goes further
— reading `environment.get_all_execution_data()`, which returns **every** node's
outputs, not just the end node's.

**Why that matters:** the earlier design had `draft_reply` echo
`final_category`/`final_priority` forward so the grader could see them. TextGrad
rewrote that instruction, the model started emitting `category`/`priority`,
pydantic rejected the output as missing required fields, the node returned `None`,
and TextGrad died with:

```
AssertionError: Value must be a string, int, or image (bytes). Got: <class 'NoneType'>
```

...25 minutes into a run. Reading all nodes removes the passthrough and with it
that entire failure class. **Do not reintroduce a passthrough.**

---

## 8. Metrics dicts must contain numbers only

`evaluator.py:327` sums every key across examples:

```python
{k: sum(d[k] for d in scores if d is not None) / n for k in first_valid_score}
```

A string value (e.g. a judge critique) gives
`TypeError: unsupported operand type(s) for +: 'int' and 'str'`.

Text diagnostics go to `judge_critiques.jsonl` instead — which is better anyway,
since an averaged critique is meaningless.

---

## 9. Declare list-valued outputs as `"type": "array"`

Declared as `"string"`, a model returning `["a","b"]` fails pydantic validation.
EvoAgentX's fallback is to fire **another LLM call** to re-extract — roughly
doubling cost on that node, silently.

`"type": "array"` accepts lists directly (verified). Watch for the
`Can not instantiate ...InputOutput... 2 validation errors` log line; it means
you're paying twice.

---

## 10. Rate limits (if you go back to Gemini)

Free tier, observed in real 429 payloads:

| Model | Limit | Quota ID |
|---|---|---|
| `gemini-3.6-flash` | 5 RPM | `GenerateRequestsPerMinutePerProjectPerModel` |
| `gemini-3.1-flash-lite` | 15 RPM | same |
| `gemini-3.6-flash` | **20/day** | `GenerateRequestsPerDayPerProjectPerModel` |

The daily cap is the killer — no client-side throttle fixes it. Also: **Pro models
were removed from the free tier in April 2026**, so any Pro optimizer/judge model
needs billing regardless of quota.

LiteLLM ignores the server's `retryDelay` hint, so it burns the minute's budget in
seconds then stalls. `rate_limit.py` gates calls through a rolling-window
semaphore; set `max_rpm` slightly *below* the real cap (Google's window is rolling
and clock skew trips exact matches).

Import `rate_limit` and call `install_rpm_limit()` **before** constructing any
LiteLLM object — it monkeypatches `litellm.completion` / `acompletion`.

---

## 11. Harmless noise you can ignore

- `AuthlibDeprecationWarning`, `PyPDF2 is deprecated` — from EvoAgentX's eager imports.
- `LiteLLM: Failed to fetch remote model cost map ... Falling back to local backup`
  — harmless in offline tests; set `LITELLM_LOCAL_MODEL_COST_MAP=True` to suppress
  the attempted refresh.
- `Task was destroyed but it is pending!`, `task_done() called too many times` —
  LiteLLM's async logging worker. Cosmetic.
- Loguru dumps every local variable at every frame on error. A single 503 produces
  ~200 lines. Read the last few lines first.

---

