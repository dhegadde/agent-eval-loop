"""Validate the LLM judge against curated cases (ROADMAP #2).

The judge in benchmark_tickets.py has been written and unit-tested but has never
been shown to behave correctly end to end. This exercises the REAL production
scoring path - TicketBenchmark._score_reply with judge_llm supplied - against a
small set of hand-built (ticket, reference, candidate) triples whose correct
verdict we know in advance, and reports whether the judge agrees.

The three questions ROADMAP #2 asks, and which cases answer them:

  - Does it flag genuine hallucinations?  -> the `flag` cases, above all
    `invented_security_policy` (the canonical Run-1 failure it MUST catch).
  - Does it wrongly flag grounded replies? -> the `clean` cases must NOT flag.
  - Is `hallucinated` ever true?            -> if no `flag` case fires, the gate
    is dead weight.

Cost: one gpt-5.4 judge call per case (a handful of cents total). This does NOT
touch the executor, the optimizer, or the dataset - it is pure judge behaviour.

    python validate_judge.py
"""

import json
from pathlib import Path

from setup_swarm import judge_llm  # constructing this reads OPENAI_API_KEY
from benchmark_tickets import TicketBenchmark

# References below are the real seed replies from data/tickets.jsonl, so the
# judge is comparing candidates against the same ground truth the eval uses.
CASES = [
    {
        "name": "grounded_refund",
        "expect": "clean",  # matches the reference's facts, invents nothing
        "ticket": "I was charged twice on 3 Aug for the same monthly subscription. Card ending 4417.",
        "reference": "Thanks for flagging this. I can see two charges dated 3 Aug on the card ending 4417 and I've issued a refund for the duplicate. It should land in 5-7 business days.",
        "candidate": "Thanks for flagging this - I can see the duplicate charge on the card ending 4417 and have refunded it. It should post back in 5-7 business days.",
    },
    {
        "name": "invented_security_policy",
        "expect": "flag",  # the canonical Run-1 hallucination; MUST be caught
        "ticket": "I was charged twice on 3 Aug for the same monthly subscription. Card ending 4417.",
        "reference": "Thanks for flagging this. I can see two charges dated 3 Aug on the card ending 4417 and I've issued a refund for the duplicate. It should land in 5-7 business days.",
        "candidate": "I've refunded the duplicate charge. For security, we won't ask for full card details - your card ending 4417 is sufficient for verification, and our fraud team has already cleared your account.",
    },
    {
        "name": "invented_timeline",
        "expect": "flag",  # a 24h engineering fix is nowhere in the ticket
        "ticket": "Can't log in. Says my password is wrong but I just reset it twice.",
        "reference": "Repeated resets can leave a stale session token. I've cleared yours - please try one more reset and log in from a private browser window. If it still fails, let me know the exact error text.",
        "candidate": "I've escalated this to our engineering team and they will have it fully fixed within 24 hours. Your account was flagged by our automated security system after the resets.",
    },
    {
        "name": "grounded_login",
        "expect": "clean",  # faithful paraphrase of the reference
        "ticket": "Can't log in. Says my password is wrong but I just reset it twice.",
        "reference": "Repeated resets can leave a stale session token. I've cleared yours - please try one more reset and log in from a private browser window. If it still fails, let me know the exact error text.",
        "candidate": "Repeated resets can leave a stale session token behind. I've cleared yours - please reset once more and sign in from a private window, and tell me the exact error text if it still fails.",
    },
    {
        "name": "restates_only",
        "expect": "clean",  # NOT a hallucination, but should score LOW (no action)
        "ticket": "Export to CSV has been spinning forever since this morning. Whole team is blocked, nobody can pull reports.",
        "reference": "Sorry about this. CSV export is failing for a subset of accounts and engineering is on it now. I'll update you the moment it's resolved. In the meantime the JSON export endpoint is unaffected if you need the data urgently.",
        "candidate": "Thanks for reaching out. I understand your CSV export has been spinning since this morning and your whole team is blocked. That sounds really frustrating.",
    },
    {
        "name": "invented_refund_amount",
        "expect": "flag",  # a specific dollar figure the ticket never states
        "ticket": "Our invoice shows 45 seats but we only have 30 people. Been overbilled for at least three months.",
        "reference": "I've checked the account and confirm 45 seats were provisioned against 30 active users. I'm correcting the seat count now and raising a credit for the last three billing cycles.",
        "candidate": "You're right - I've refunded $1,240 to your account today and downgraded you to 30 seats effective immediately. The refund will appear on your statement within 2 business days.",
    },
]


def main():
    # judge_llm supplied -> _score_reply uses the real judge, not the heuristic.
    bench = TicketBenchmark(judge_llm=judge_llm)

    out_path = Path("judge_validation.jsonl")
    out_path.write_text("")  # fresh each run

    rows = []
    for c in CASES:
        score, hallucinated, critique = bench._score_reply(
            ticket=c["ticket"], reference=c["reference"], candidate=c["candidate"]
        )
        # A `flag` case passes if the gate fired; a `clean` case passes if it did not.
        passed = (hallucinated is True) if c["expect"] == "flag" else (hallucinated is False)
        row = {
            "name": c["name"], "expect": c["expect"],
            "hallucinated": hallucinated, "score": round(score, 3),
            "pass": passed, "critique": critique,
        }
        rows.append(row)
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # ---- report ----
    print(f"\n{'case':<26} {'expect':<7} {'halluc':<7} {'score':<6} {'verdict':<6} critique")
    print("-" * 100)
    for r in rows:
        verdict = "PASS" if r["pass"] else "FAIL"
        print(f"{r['name']:<26} {r['expect']:<7} {str(r['hallucinated']):<7} "
              f"{r['score']:<6} {verdict:<6} {r['critique']}")

    n_pass = sum(r["pass"] for r in rows)
    print("-" * 100)
    print(f"{n_pass}/{len(rows)} cases behaved as expected.")

    # The specific diagnostics ROADMAP #2 asks for.
    clean = [r for r in rows if r["expect"] == "clean"]
    flag = [r for r in rows if r["expect"] == "flag"]
    canonical = next(r for r in rows if r["name"] == "invented_security_policy")
    print("\nDiagnostics:")
    print(f"  canonical Run-1 hallucination caught: {canonical['hallucinated'] is True}")
    print(f"  false positives on grounded replies : {sum(r['hallucinated'] is True for r in clean)}")
    print(f"  hallucination gate ever fires        : {any(r['hallucinated'] is True for r in flag)}")
    grounded_scores = [r['score'] for r in rows if r['name'] in ('grounded_refund', 'grounded_login')]
    restates = next(r for r in rows if r['name'] == 'restates_only')
    print(f"  grounded replies score high          : {grounded_scores} (want >~0.6)")
    print(f"  content-free 'restates_only' scores   : {restates['score']} (want low, and NOT flagged)")
    print(f"\nfull rows -> {out_path}")


if __name__ == "__main__":
    main()
