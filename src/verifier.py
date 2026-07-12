"""
Staleness Verifier — the single LLM call proven out in Milestone 1, now
refactored into an importable, reusable function instead of a standalone
script.
"""

import json
import os

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


def _build_client() -> OpenAI:
    api_key = os.environ["LLM_API_KEY"]
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def judge_staleness(old_code: str, new_code: str, doc_section: str, model: str) -> dict:
    """Returns {"stale": bool|None, "diagnosis": str}.

    stale=None signals a failure (LLM error or unparseable response) — the
    caller is responsible for fail-open handling, this function never raises.
    """
    client = _build_client()
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
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"stale": None, "diagnosis": f"[LLM CALL FAILED: {e}]"}

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
