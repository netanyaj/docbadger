"""
DocBadger — Milestone 1 Spike
Proves (or disproves) the core assumption: can an LLM reliably judge whether
a documentation section is stale relative to a code change?

This script is intentionally throwaway-quality. No persistence, no GitHub
integration, no rubric — just the single riskiest LLM call, run against a
handful of hand-authored test cases, so we can eyeball the results before
building anything else around it.

Usage:
    export OPENROUTER_API_KEY="..."
    python staleness_spike.py --model openai/gpt-4o
    python staleness_spike.py --model anthropic/claude-sonnet-4.5

Check https://openrouter.ai/models for the exact current model slugs —
these change over time, so don't take the examples above as guaranteed-current.
"""

import argparse
import json
import os
import sys

from openai import OpenAI

SYSTEM_PROMPT = """You are a documentation accuracy auditor. You will be shown:
1. The OLD version of a code function/class.
2. The NEW version of that same function/class, after a change.
3. A documentation section that describes this code's behavior.

Your job: determine whether the documentation is now STALE — meaning it no
longer accurately describes the NEW code's behavior.

Respond with ONLY a JSON object, no other text, no markdown fences, in this
exact shape:
{
  "stale": true or false,
  "diagnosis": "one or two sentences explaining your reasoning, specific to
                 what changed and why it does or doesn't affect the doc"
}

Be precise. Do not flag a section as stale just because the code changed —
only flag it if the change actually contradicts or invalidates something the
documentation claims. If the documentation is still technically accurate,
even if incomplete, lean towards NOT stale and say so in your diagnosis.
"""

USER_PROMPT_TEMPLATE = """OLD CODE:
```
{old_code}
```

NEW CODE:
```
{new_code}
```

DOCUMENTATION SECTION:
```
{doc_section}
```

Is the documentation section now stale relative to the new code?"""


def build_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def judge_staleness(client: OpenAI, model: str, old_code: str, new_code: str, doc_section: str) -> dict:
    """Makes the single LLM call this whole milestone exists to validate.

    Deliberately minimal error handling here (this is a spike) — but note
    that even at this stage we're already anticipating the fail-open
    principle from the architecture doc: a malformed/failed response
    becomes a clearly-labeled error result, not a crash.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        old_code=old_code, new_code=new_code, doc_section=doc_section
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=500,  # our expected output is a couple sentences of JSON;
                              # capping this avoids provider default request-budget
                              # errors on limited-credit accounts (see Milestone 1
                              # troubleshooting note in Engineering Decision Log).
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"stale": None, "diagnosis": f"[LLM CALL FAILED: {e}]"}

    # Models occasionally wrap JSON in markdown fences despite instructions —
    # strip that defensively rather than assume perfect compliance.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return {"stale": parsed.get("stale"), "diagnosis": parsed.get("diagnosis", "")}
    except json.JSONDecodeError:
        return {"stale": None, "diagnosis": f"[UNPARSEABLE RESPONSE: {raw[:200]}]"}


def main():
    parser = argparse.ArgumentParser(description="DocBadger Milestone 1 staleness-judgment spike")
    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="OpenRouter model slug, e.g. openai/gpt-4o or anthropic/claude-sonnet-4.5",
    )
    parser.add_argument(
        "--cases",
        default="test_cases.json",
        help="Path to the hand-authored test cases JSON file",
    )
    args = parser.parse_args()

    with open(args.cases) as f:
        cases = json.load(f)

    client = build_client()

    print(f"\nRunning {len(cases)} test cases against model: {args.model}\n")
    print("=" * 80)

    agree_count = 0
    total_judged = 0

    for case in cases:
        result = judge_staleness(
            client, args.model, case["old_code"], case["new_code"], case["doc_section"]
        )

        expected = case["expected_label"]
        actual = "stale" if result["stale"] is True else ("not_stale" if result["stale"] is False else "error")

        # "ambiguous" expected cases don't count against agreement either way —
        # they're there to observe behavior, not to grade pass/fail.
        agreement_marker = ""
        if expected != "ambiguous" and actual != "error":
            total_judged += 1
            if (expected == "stale" and actual == "stale") or (expected == "not_stale" and actual == "not_stale"):
                agree_count += 1
                agreement_marker = "✅ MATCH"
            else:
                agreement_marker = "❌ MISMATCH"

        print(f"[{case['id']}]  expected={expected}  model_said={actual}  {agreement_marker}")
        print(f"  Diagnosis: {result['diagnosis']}")
        print(f"  (Why this case exists: {case['why']})")
        print("-" * 80)

    print("=" * 80)
    if total_judged > 0:
        print(f"\nAgreement on clear-cut cases: {agree_count}/{total_judged}")
    print("Review the 'ambiguous' cases manually above — there's no right answer to grade,")
    print("only whether the model's reasoning seems sound to a human reader.\n")


if __name__ == "__main__":
    main()
