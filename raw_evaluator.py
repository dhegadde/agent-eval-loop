"""Evaluator that grades the workflow's raw output dict instead of an LLM summary.

Why this exists:

EvoAgentX's stock Evaluator calls `workflow.execute(..., extract_output=True)`,
which fires an EXTRA LLM call to summarize the run into a prose string, and that
string is what gets graded. Two problems:

  1. Correctness. The end node emits a clean dict
     {"final_category": "billing", "final_priority": "P1", "reply": "..."} and the
     summarizer throws the structure away, so category/priority score 0.0 even
     when the agents got them right.
  2. Cost. It is one extra call per example per evaluation - about 25% of your
     request budget, which matters a lot on a rate-limited free tier.

`extract_output=False` returns the declared outputs verbatim and free. We go one
step further and read `environment.get_all_execution_data()`, which holds the
outputs of EVERY node.

Why that matters: previously `draft_reply` had to echo `final_category` /
`final_priority` forward so the grader could see them. TextGrad rewrote that
instruction, the model started emitting `category` / `priority` instead, pydantic
rejected the output as missing required fields, the node returned None, and
TextGrad crashed with "Value must be a string, int, or image (bytes). Got:
<class 'NoneType'>". Reading all nodes removes the passthrough - and with it that
entire class of failure.

The stock Evaluator hardcodes extract_output=True, so we override the one method.
"""

from typing import List, Tuple, Union

from evoagentx.core.message import Message
from evoagentx.evaluators import Evaluator
from evoagentx.workflow import WorkFlow, WorkFlowGraph


class RawOutputEvaluator(Evaluator):
    def _execute_workflow_graph(
        self,
        graph: WorkFlowGraph,
        inputs: dict,
        return_trajectory: bool = False,
        **kwargs,
    ) -> Union[dict, Tuple[dict, List[Message]]]:
        if self.agent_manager is None:
            raise ValueError("`agent_manager` is required to evaluate a WorkFlowGraph.")

        graph_copy = WorkFlowGraph(goal=graph.goal, graph=graph)
        graph_copy.reset_graph()
        workflow = WorkFlow(
            llm=self.llm,
            graph=graph_copy,
            agent_manager=self.agent_manager,
            **kwargs,
        )

        # extract_output=False -> raw dict, no summarizer call.
        result = workflow.execute(inputs=inputs, extract_output=False, **kwargs)

        if result.status == "success":
            # Pull EVERY node's outputs, not just the end node's. This is what lets
            # `category`/`priority` be read straight off the triage node, so the
            # draft node never has to echo them forward. See the note below.
            output = dict(workflow.environment.get_all_execution_data())
        else:
            output = {"error": result.displayable_error or "Workflow Execution Failed"}

        if return_trajectory:
            return output, workflow.environment.get()
        return output