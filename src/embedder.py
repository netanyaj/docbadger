"""
Embedder — thin wrapper around the configured provider's embeddings
endpoint (see llm_client.py for provider selection via LLM_PROVIDER).
Deliberately separated from embedding_linker.py's similarity logic so that
logic can be tested with fake vectors, with no real API calls needed.
"""

from llm_client import _build_client, call_with_rate_limit_retry, current_provider, DEFAULT_EMBEDDING_MODEL


def embed_texts(texts: list[str], model: str = None) -> list[list[float]]:
    """Batch-embeds a list of texts in one API call. Returns an empty list
    immediately for empty input, without making a network call.

    model defaults to the current provider's default embedding model
    (LLM_PROVIDER-aware) rather than a single hardcoded OpenRouter model
    string, since providers don't share an embedding-model catalog."""
    if not texts:
        return []
    if model is None:
        model = DEFAULT_EMBEDDING_MODEL[current_provider()]
    client = _build_client()
    response = call_with_rate_limit_retry(
        lambda: client.embeddings.create(model=model, input=texts)
    )
    return [item.embedding for item in response.data]
