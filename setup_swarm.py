"""Provider-neutral model wiring + the reference swarm definition.

Two things here are load-bearing and easy to get wrong:

1. `parse_mode: "json"` on every task.
   SequentialWorkFlowGraph defaults parse_mode to "str", which assigns the ENTIRE
   raw response to every declared output field. That is why an unconfigured run
   scores 0.0: `category` and `priority` both end up holding the whole blob.

2. `prompt_template` on every task, and NO `prompt` key.
   TextGradOptimizer reads `agent.actions[0].prompt_template.instruction`. If you
   only supply `prompt`, that attribute is None and optimization dies with
   AttributeError before the first step.

   If you supply BOTH, CustomizeAgent logs a warning and silently ignores `prompt`
   entirely - so any output-format rules you wrote there never reach the model.
   Don't bother: with parse_mode="json" the template auto-generates the JSON
   schema instruction from the `outputs` descriptions below. That is why each
   output's `description` matters - it is rendered into the prompt verbatim.

   The `instruction` string is what TextGrad actually rewrites, so the real task
   guidance belongs there.

Provider settings are loaded independently for executor, optimizer, and judge by
``model_runtime.py``.  Defaults remain OpenAI; an explicitly configured
``openai_compatible`` role can point at a private vLLM endpoint.
"""

import litellm
from dotenv import load_dotenv

from evoagentx.prompts.template import StringTemplate
from evoagentx.workflow import SequentialWorkFlowGraph

from model_runtime import build_role_runtime, runtime_manifest

load_dotenv()

# Transient 429/503s happen on any provider. Optimization makes hundreds of calls,
# so without retries a single spike kills a run partway through.
litellm.num_retries = 5

# Each role is independently swappable.  Building three runtime objects keeps the
# judge a separate instance even when two roles happen to use the same model.
_executor_runtime = build_role_runtime("executor")
_optimizer_runtime = build_role_runtime("optimizer")
_judge_runtime = build_role_runtime("judge")

executor_config = _executor_runtime.config
executor_llm = _executor_runtime.llm
optimizer_config = _optimizer_runtime.config
optimizer_llm = _optimizer_runtime.llm
judge_config = _judge_runtime.config
judge_llm = _judge_runtime.llm

MODEL_RUNTIME_MANIFEST = runtime_manifest(
    _executor_runtime.settings,
    _optimizer_runtime.settings,
    _judge_runtime.settings,
)


GOAL = (
    "Triage an inbound customer support ticket and draft the reply that will be "
    "sent to the customer."
)

CATEGORY_LIST = "billing, bug, account_access, feature_request, other"
PRIORITY_LIST = "P0, P1, P2, P3"


TASKS = [
    {
        "name": "triage",
        "description": "Classify the ticket into a category and an urgency priority.",
        "inputs": [
            {"name": "ticket", "type": "string", "required": True,
             "description": "The raw inbound customer ticket text."},
        ],
        "outputs": [
            {"name": "category", "type": "string", "required": True,
             "description": f"One of: {CATEGORY_LIST}."},
            {"name": "priority", "type": "string", "required": True,
             "description": f"One of: {PRIORITY_LIST}."},
        ],
        "parse_mode": "json",
        "prompt_template": StringTemplate(
            instruction=(
                "Read the support ticket and classify it. Choose exactly one "
                f"category from: {CATEGORY_LIST}. Choose exactly one priority "
                f"from: {PRIORITY_LIST}. Judge priority by business impact: "
                "production breakage or many users blocked is P0, individual "
                "users blocked or money incorrectly taken is P1, and requests "
                "with no time pressure are P3."
            )
        ),
    },
    {
        "name": "extract_facts",
        "description": "Pull the concrete details needed to answer, and note what is missing.",
        "inputs": [
            {"name": "ticket", "type": "string", "required": True,
             "description": "The raw inbound customer ticket text."},
            {"name": "category", "type": "string", "required": True,
             "description": "The category assigned by triage."},
        ],
        "outputs": [
            {"name": "facts", "type": "array", "required": True,
             "description": "Short factual bullets drawn only from the ticket."},
            {"name": "missing", "type": "array", "required": True,
             "description": "Details the customer did not provide but that are needed."},
        ],
        "parse_mode": "json",
        "prompt_template": StringTemplate(
            instruction=(
                "List only facts stated explicitly in the ticket. Do not infer, "
                "assume, or invent anything, and do not restate the assigned "
                "category or priority as if they were facts from the customer. "
                "Then list the details a support agent would need in order to "
                "resolve this but that the customer did not provide."
            )
        ),
    },
    {
        "name": "draft_reply",
        "description": "Write the customer-facing reply.",
        "inputs": [
            {"name": "ticket", "type": "string", "required": True,
             "description": "The raw inbound customer ticket text."},
            {"name": "category", "type": "string", "required": True,
             "description": "The category assigned by triage."},
            {"name": "priority", "type": "string", "required": True,
             "description": "The priority assigned by triage."},
            {"name": "facts", "type": "array", "required": True,
             "description": "Facts extracted from the ticket."},
            {"name": "missing", "type": "array", "required": True,
             "description": "Details the customer did not provide."},
        ],
        "outputs": [
            {"name": "reply", "type": "string", "required": True,
             "description": "The final reply to send to the customer."},
        ],
        "parse_mode": "json",
        "prompt_template": StringTemplate(
            instruction=(
                "Write a short, direct reply to the customer. Assert nothing that "
                "is not in the supplied facts - never invent refunds, timelines, "
                "account details, or root causes. If details needed to resolve the "
                "issue are missing, ask for them plainly. Do not open with filler "
                "pleasantries or restate the ticket back to the customer."
            )
        ),
    },
]


def build_graph() -> SequentialWorkFlowGraph:
    return SequentialWorkFlowGraph(goal=GOAL, tasks=TASKS)
