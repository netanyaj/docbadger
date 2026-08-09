"""
Prompt Versioning -- Architecture Section 16: "Prompt versioning is
mandatory, not optional -- every prompt change gets a new version ID, and
the eval harness (Section 19) records which prompt version produced which
precision/recall numbers."

Each LLM-calling stage (verifier, corrector, validator) declares its own
PROMPT_VERSION string constant. A manually-maintained version string can
silently drift from the prompt text it's meant to identify -- a developer
edits a prompt's wording and simply forgets to bump the constant -- which
is exactly the failure mode this section exists to prevent ("we cannot
answer 'did this prompt change make things better or worse'").

hash_prompt_pair() is the shared, single source of truth behind a small
mechanical safety net used by each stage's own test suite: a "golden hash"
test calls the stage's _build_prompts() with a fixed nonce and asserts the
hash of the rendered (system_prompt, user_prompt) matches a stored expected
value. If the prompt text changes without a matching version bump, that
test fails loudly -- versioning stays mandatory by construction (a failing
test), not by developer memory.
"""

import hashlib


def hash_prompt_pair(system_prompt: str, user_prompt: str) -> str:
    combined = "\x00".join([system_prompt, user_prompt])
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
