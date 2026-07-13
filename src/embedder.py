"""
Embedder — thin wrapper around OpenRouter's embeddings endpoint.
Deliberately separated from embedding_linker.py's similarity logic so that
logic can be tested with fake vectors, with no real API calls needed.
"""

import os

from openai import OpenAI


def _build_client() -> OpenAI:
    api_key = os.environ["LLM_API_KEY"]
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def embed_texts(texts: list[str], model: str = "openai/text-embedding-3-small") -> list[list[float]]:
    """Batch-embeds a list of texts in one API call. Returns an empty list
    immediately for empty input, without making a network call."""
    if not texts:
        return []
    client = _build_client()
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
