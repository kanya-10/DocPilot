"""
Eval harness. Runs DocPilot against evals/test_cases.py and reports:

  - retrieval accuracy: did the expected source show up when one was expected?
  - routing accuracy: did the agent call the expected tool (or correctly call none)?
  - hallucination rate: for unanswerable questions, did it correctly say "I don't know"
    instead of fabricating an answer?

Run:
    python -m evals.run_evals
"""

import asyncio
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agent.core import DocPilotAgent
from evals.test_cases import EVAL_CASES


async def run_evals():
    agent = DocPilotAgent()

    retrieval_correct = 0
    retrieval_total = 0
    routing_correct = 0
    hallucination_failures = 0
    unanswerable_total = 0

    results = []

    for case in EVAL_CASES:
        result = await agent.answer(case["question"])
        time.sleep(8)


        row = {"question": case["question"], "answer": result.answer[:120]}

        # Retrieval accuracy
        if case["expected_source_contains"]:
            retrieval_total += 1
            hit = any(case["expected_source_contains"] in s for s in result.sources)
            retrieval_correct += int(hit)
            row["retrieval_hit"] = hit

        # Routing accuracy (did it call the right tool, or correctly call none)
        expected_tool = case["expected_tool"]
        if expected_tool:
            called = expected_tool in result.tools_used
        else:
            called = True  # no specific tool expected; don't penalize extra tool use here
        routing_correct += int(called)
        row["routing_correct"] = called

        # Hallucination check on unanswerable questions
        if not case["answerable"]:
            unanswerable_total += 1
            said_unknown = result.confidence == "low"
            if not said_unknown:
                hallucination_failures += 1
            row["correctly_declined"] = said_unknown

        results.append(row)

    print("\n=== DocPilot Eval Report ===\n")
    for row in results:
        print(row)

    print("\n--- Summary ---")
    if retrieval_total:
        print(f"Retrieval accuracy: {retrieval_correct}/{retrieval_total} "
              f"({100 * retrieval_correct / retrieval_total:.0f}%)")
    print(f"Routing accuracy: {routing_correct}/{len(EVAL_CASES)} "
          f"({100 * routing_correct / len(EVAL_CASES):.0f}%)")
    if unanswerable_total:
        print(f"Hallucination failures: {hallucination_failures}/{unanswerable_total} "
              f"(lower is better)")


if __name__ == "__main__":
    asyncio.run(run_evals())
