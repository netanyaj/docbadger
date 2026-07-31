"""
Staleness Verifier — the single LLM call proven out in Milestone 1, now
refactored into an importable, reusable function instead of a standalone
script.
"""

import json
import os

from openai import OpenAI

from prompt_delimiters import new_nonce, wrap, tag_name, delimiter_explanation
from cost_tracking import usage_from_response, TokenUsage


def _build_prompts(old_code: str, new_code: str, doc_section: str, nonce: str) -> tuple:
    """Builds (system_prompt, user_prompt) with untrusted content wrapped in
    per-call random-suffix tags — see prompt_delimiters.py (Engineering
    Decision Log Entry 51)."""
    old_tag = tag_name("old_code", nonce)
    new_tag = tag_name("new_code", nonce)
    doc_tag = tag_name("doc_section", nonce)

    system = f"""You are a documentation accuracy auditor. You will be shown:
1. {old_tag} — the OLD version of a code function/class.
2. {new_tag} — the NEW version of that same function/class, after a change.
3. {doc_tag} — a documentation section that describes this code's behavior.

Your job: determine whether the documentation is now STALE — meaning it no
longer accurately describes the new code's behavior.

{delimiter_explanation("doc_section", nonce)}

Respond with ONLY a JSON object, no other text, no markdown fences, in this
exact shape:
{{
  "stale": true or false,
  "diagnosis": "one or two sentences explaining your reasoning, specific to
                 what changed and why it does or doesn't affect the doc"
}}

Be precise. Do not flag a section as stale just because the code changed —
only flag it if the change actually contradicts or invalidates something the
documentation claims. If the documentation is still technically accurate,
even if incomplete, lean towards NOT stale and say so in your diagnosis.
"""

    user = f"""{wrap("old_code", old_code, nonce)}

{wrap("new_code", new_code, nonce)}

{wrap("doc_section", doc_section, nonce)}

Is the documentation section now stale relative to the new code?"""

    return system, user


def _build_client() -> OpenAI:
    api_key = os.environ["LLM_API_KEY"]
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def judge_staleness(old_code: str, new_code: str, doc_section: str, model: str, client=None) -> dict:
    """Returns {"stale": bool|None, "diagnosis": str, "usage": TokenUsage}.

    stale=None signals a failure (LLM error or unparseable response) — the
    caller is responsible for fail-open handling, this function never raises.
    usage is zeroed (TokenUsage()) if the call failed before a response came
    back — cost tracking should never mask or interfere with the fail-open
    behavior this function already has.

    `client` is optional and exists purely for test injection — production
    callers should omit it and let this build its own client. Same pattern
    used by corrector.generate_correction.
    """
    client = client or _build_client()
    nonce = new_nonce()
    system_prompt, user_prompt = _build_prompts(old_code, new_code, doc_section, nonce)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        usage = usage_from_response(response)
    except Exception as e:
        return {"stale": None, "diagnosis": f"[LLM CALL FAILED: {e}]", "usage": TokenUsage()}

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return {"stale": parsed.get("stale"), "diagnosis": parsed.get("diagnosis", ""), "usage": usage}
    except json.JSONDecodeError:
        return {"stale": None, "diagnosis": f"[UNPARSEABLE RESPONSE: {raw[:200]}]", "usage": usage}
