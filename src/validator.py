"""
Validator stage — DocBadger pipeline.

Independently checks a Corrector-proposed patch (old_text/new_text) before it's
allowed into a PR. Only ever called on Corrector results with
status == "proposed" — Corrector abstentions never reach this module.

Critical design constraint (Product Decision Log Entry 6): this module is
deliberately NEVER given the Verifier's diagnosis. If it saw the diagnosis, a
subtly-wrong diagnosis could anchor the Validator's judgment the same way it
may have anchored the Corrector's — the Validator would stop being an
independent check and become "the same reasoning, asked twice." Its only
inputs are new_code (ground truth), the original doc_section, and the
proposed old_text/new_text patch itself.

Three checks, cheapest first (same "deterministic before probabilistic"
ordering used everywhere else in this pipeline — Change Filter before the
Verifier, heuristic linking before embedding linking):

  1. Structural  — deterministic, no LLM call. Re-verifies old_text is still
                    an exact substring of doc_section (defense in depth — the
                    Corrector already checked this, cheap to recheck) and
                    lints the patched result for broken Markdown mechanics.
  2. Accuracy    — LLM call. Does new_text actually correctly describe new_code?
  3. Style       — same LLM call as accuracy (both are "read this and assess
                    it" against the same inputs, unlike Verifier/Corrector,
                    which need genuinely different framings).

Per product decision (this session): ANY rejection — structural, accuracy, or
style — routes to comment-only for v1, not just accuracy failures. The
proposed old_text/new_text is still surfaced in the comment either way, so a
style-only rejection still gives the reviewer the drafted text to accept or
polish by hand — the rejection means "we won't auto-propose this in the PR
diff," not "we withhold it from you entirely."
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from verifier import _build_client


class ValidationStatus(str, Enum):
    APPROVED = "approved"
    REJECTED_STRUCTURAL = "rejected_structural"
    REJECTED_ACCURACY = "rejected_accuracy"
    REJECTED_STYLE = "rejected_style"
    ERROR_UNVALIDATED = "error_unvalidated"   # LLM responded, but not in a usable shape
    ERROR_INFRA = "error_infra"                # the LLM call itself raised


@dataclass
class ValidatorResult:
    status: ValidationStatus
    old_text: str      # echoed back unchanged — always present, for the comment to render
    new_text: str       # echoed back unchanged — always present, for the comment to render
    rationale: str       # always populated: why approved, or why rejected/unvalidated


SYSTEM_PROMPT = """You are the Validator stage in a documentation-staleness pipeline.
You are an INDEPENDENT check — you have not been told why an earlier stage
believed this documentation needed a correction, and you should not assume
its reasoning was right. Judge only what is in front of you.

You will be shown:
1. NEW CODE (ground truth) — the actual current code.
2. ORIGINAL DOC SECTION — the full documentation section before any change.
3. PROPOSED OLD TEXT — the exact span of the original section a Corrector
   stage wants to replace.
4. PROPOSED NEW TEXT — what it wants to replace that span with.

Assess two independent things:
- ACCURACY: does PROPOSED NEW TEXT correctly and completely describe what
  NEW CODE actually does? Flag anything invented, dropped, or wrong.
- STYLE: does PROPOSED NEW TEXT read naturally alongside the untouched parts
  of ORIGINAL DOC SECTION — consistent tone, tense, and terminology? A
  correction can be factually accurate and still read awkwardly against its
  surroundings; that is a style problem, not an accuracy problem.

If NEW CODE does not clearly support the proposed text as accurate, reject
on accuracy even if it reads well. Only reject on style if accuracy is fine
but the phrasing genuinely clashes with the surrounding prose.

Respond with ONLY a JSON object, no other text, no markdown fences, in
exactly this shape:
{"status": "approved" | "rejected_accuracy" | "rejected_style", "rationale": "<one to two sentences, specific to what you checked>"}
"""

USER_PROMPT_TEMPLATE = """NEW CODE:
```
{new_code}
```

ORIGINAL DOC SECTION:
```
{doc_section}
```

PROPOSED OLD TEXT:
```
{old_text}
```

PROPOSED NEW TEXT:
```
{new_text}
```

Assess accuracy and style."""


def _check_structural(doc_section: str, old_text: str, new_text: str) -> Optional[str]:
    """Deterministic checks only. Returns a rejection reason string if a check
    fails, or None if the patch passes. Intentionally heuristic-level (brace/
    fence balance), not a full Markdown parser — consistent with this
    project's v1 scoping elsewhere (e.g. the heuristic linker)."""
    if old_text not in doc_section:
        return "old_text is not present verbatim in doc_section (re-check failed)."

    patched = doc_section.replace(old_text, new_text, 1)

    if new_text.count("`") % 2 != 0:
        return "Proposed new_text has an unbalanced number of backticks."

    if patched.count("```") % 2 != 0:
        return "Patched section has an unbalanced number of code-fence markers (```)."

    if new_text.count("[") != new_text.count("]"):
        return "Proposed new_text has unbalanced square brackets (possible broken Markdown link)."

    if new_text.count("(") != new_text.count(")"):
        return "Proposed new_text has unbalanced parentheses (possible broken Markdown link)."

    return None


def _call_llm(user_prompt: str, model: str, client) -> str:
    """Raises on API failure — caller handles fail-open, same division of
    responsibility as verifier.judge_staleness and corrector.generate_correction."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _parse_response(raw: str) -> dict:
    data = json.loads(raw)  # may raise json.JSONDecodeError — caller handles it
    status = data.get("status")
    if status not in {"approved", "rejected_accuracy", "rejected_style"}:
        raise ValueError(f"Unrecognized status in Validator response: {status!r}")
    if not data.get("rationale"):
        raise ValueError("Validator response missing rationale")
    return data


def validate_correction(
    new_code: str,
    doc_section: str,
    old_text: str,
    new_text: str,
    model: str,
    client=None,
) -> ValidatorResult:
    """Only call this on Corrector results with status == "proposed".
    `client` is optional and exists purely for test injection, same pattern
    as corrector.generate_correction and verifier.judge_staleness.
    """
    structural_failure = _check_structural(doc_section, old_text, new_text)
    if structural_failure:
        return ValidatorResult(
            status=ValidationStatus.REJECTED_STRUCTURAL,
            old_text=old_text,
            new_text=new_text,
            rationale=structural_failure,
        )

    client = client or _build_client()
    user_prompt = USER_PROMPT_TEMPLATE.format(
        new_code=new_code, doc_section=doc_section, old_text=old_text, new_text=new_text
    )

    try:
        raw = _call_llm(user_prompt, model, client)
    except Exception as e:
        return ValidatorResult(
            status=ValidationStatus.ERROR_INFRA,
            old_text=old_text,
            new_text=new_text,
            rationale=f"[LLM CALL FAILED: {e}] — proposal not validated, treat with caution.",
        )

    try:
        data = _parse_response(raw)
    except (json.JSONDecodeError, ValueError) as e:
        return ValidatorResult(
            status=ValidationStatus.ERROR_UNVALIDATED,
            old_text=old_text,
            new_text=new_text,
            rationale=f"Validator response could not be parsed: {e} — proposal not validated, treat with caution.",
        )

    status_map = {
        "approved": ValidationStatus.APPROVED,
        "rejected_accuracy": ValidationStatus.REJECTED_ACCURACY,
        "rejected_style": ValidationStatus.REJECTED_STYLE,
    }
    return ValidatorResult(
        status=status_map[data["status"]],
        old_text=old_text,
        new_text=new_text,
        rationale=data["rationale"],
    )
