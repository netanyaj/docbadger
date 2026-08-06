"""
LLM Client — centralized OpenAI-SDK-compatible client construction, shared
by every LLM-calling stage (verifier, corrector, validator, embedder).

Provider is selected via LLM_PROVIDER (default: "openrouter", unchanged
production behavior when unset). This exists so a provider swap is a one-
line env var change instead of a rewrite, and so client-construction logic
isn't duplicated across four call sites -- the same "don't implement
security/infra-relevant logic three separate times" reasoning as
prompt_delimiters.py (Engineering Decision Log Entry 51). Before this,
_build_client() was defined independently in verifier.py AND embedder.py.

IMPORTANT, per Entry 7: the v1 default LLM (GPT-4o) was chosen after a
real, if limited (n=6), head-to-head comparison against Claude Sonnet on
OpenRouter -- it has NOT been compared against Gemini in any documented
run. LLM_PROVIDER=gemini is here to support (a) cost-constrained testing,
e.g. the Milestone 7 real-repo dry run, where Gemini's much larger context
window avoids the mega-doc-section cost problem entirely, and (b) actually
RUNNING a real Gemini-vs-GPT-4o comparison via scripts/run_eval.py against
the 30-case eval dataset. It should not be treated as a validated
replacement for the production default (LLM_MODEL's own default stays
"openai/gpt-4o" via OpenRouter) until that comparison happens and is
logged -- the same evidence standard Entry 7 itself was held to.

This module also centralizes rate-limit handling (call_with_rate_limit_
retry / _pace), added after real 429s were hit running the Milestone 7
dry run against a real repo (small, curated eval cases never got close to
any provider's per-minute limits; a real repo's fan-out of Verifier/
Corrector/Validator calls does). Every LLM-calling stage routes its actual
network call through call_with_rate_limit_retry so this is handled once,
not reimplemented per stage.
"""

import os
import random
import sys
import time
from typing import Callable

from openai import OpenAI, RateLimitError

_PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "LLM_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
    },
}

# Provider-appropriate embedding model defaults -- OpenRouter and Gemini use
# different naming/catalogs, so embed_texts() can't share one hardcoded
# default across providers the way chat models already do via LLM_MODEL.
DEFAULT_EMBEDDING_MODEL = {
    "openrouter": "openai/text-embedding-3-small",
    "gemini": "gemini-embedding-001",
}

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY_SECONDS = 2.0

_last_call_monotonic = 0.0


def current_provider() -> str:
    return os.environ.get("LLM_PROVIDER", "openrouter")


def required_api_key_env() -> str:
    """The env var name this process needs set, given the current
    LLM_PROVIDER -- used by callers (e.g. scripts/run_eval.py) that want to
    fail with a clear, provider-aware message before spending anything,
    rather than hardcoding a single provider's key name."""
    provider = current_provider()
    return _PROVIDERS.get(provider, _PROVIDERS["openrouter"])["api_key_env"]


def _build_client() -> OpenAI:
    provider = current_provider()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}' -- supported: {sorted(_PROVIDERS)}"
        )
    config = _PROVIDERS[provider]
    api_key = os.environ[config["api_key_env"]]
    return OpenAI(base_url=config["base_url"], api_key=api_key)


def _retry_after_seconds(err: RateLimitError):
    """Prefer the API's own Retry-After header over a guessed backoff --
    it's authoritative when present."""
    try:
        header = err.response.headers.get("retry-after")
        return float(header) if header is not None else None
    except Exception:
        return None


def _pace() -> None:
    """Optional proactive throttle: if LLM_MIN_INTERVAL_SECONDS is set,
    sleeps as needed to keep consecutive calls at least that far apart.
    Off by default (0) -- exists for a caller who knows their tier's RPM
    ahead of time and would rather stay under it than react to 429s after
    the fact. Complementary to, not a replacement for, the retry-on-429
    handling below, since the exact limit for a given project/tier isn't
    knowable from code alone (see Gemini's per-project, tier-dependent
    limits — https://ai.google.dev/gemini-api/docs/rate-limits)."""
    global _last_call_monotonic
    min_interval = float(os.environ.get("LLM_MIN_INTERVAL_SECONDS", "0"))
    if min_interval <= 0:
        return
    now = time.monotonic()
    wait = _last_call_monotonic + min_interval - now
    if wait > 0:
        time.sleep(wait)
    _last_call_monotonic = time.monotonic()


def call_with_rate_limit_retry(fn: Callable, max_retries: int = None, base_delay: float = None):
    """Calls fn() (a zero-arg callable wrapping one real API call) and
    transparently retries on a 429 rate-limit response with exponential
    backoff + jitter, honoring the API's own Retry-After header when
    present. Any OTHER exception propagates immediately, unretried -- same
    "don't retry what retrying can't fix" discipline as index_branch_sync.
    push_file's non-fast-forward handling (Engineering Decision Log Entry
    63): a rate limit is transient and worth waiting out; an auth, schema,
    or network error is not, and retrying it would only obscure the real
    cause and burn the retry budget for nothing.

    max_retries / base_delay default to LLM_MAX_RETRIES / LLM_RETRY_BASE_
    DELAY_SECONDS env vars (falling back to sane hardcoded defaults) so a
    caller under a known tight quota can tune backoff without a code
    change.
    """
    max_retries = (
        max_retries if max_retries is not None
        else int(os.environ.get("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    )
    base_delay = (
        base_delay if base_delay is not None
        else float(os.environ.get("LLM_RETRY_BASE_DELAY_SECONDS", DEFAULT_BASE_DELAY_SECONDS))
    )

    attempt = 0
    while True:
        _pace()
        try:
            return fn()
        except RateLimitError as e:
            attempt += 1
            if attempt > max_retries:
                raise
            retry_after = _retry_after_seconds(e)
            delay = retry_after if retry_after is not None else base_delay * (2 ** (attempt - 1))
            delay += random.uniform(0, 0.5 * delay)  # jitter -- avoid synchronized retry storms
            print(
                f"[llm_client] Rate limited (attempt {attempt}/{max_retries}) -- "
                f"waiting {delay:.1f}s before retry.",
                file=sys.stderr,
            )
            time.sleep(delay)
