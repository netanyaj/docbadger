"""
Manual sanity check for the embedding linker — NOT part of the automated
test suite (embedder.py deliberately has no automated tests, see Milestone
3 discussion). Run this by hand, once, to confirm real embeddings behave
sensibly before trusting the pipeline to rely on them.

Usage:
    export LLM_API_KEY="your-openrouter-key"   # same key used elsewhere
    python scripts/embedding_sanity_check.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedder import embed_texts
from embedding_linker import cosine_similarity

# The exact "behavior described without naming the function" case from our
# Milestone 1 test set — the whole reason the embedding fallback needs to
# exist at all, since no name-based heuristic could ever catch this pair.
CODE_RELATED = """
def process_payment(order):
    if order.total > FRAUD_THRESHOLD:
        flag_for_review(order)
        return None
    charge_card(order.card, order.total)
    return Receipt(order)
"""

DOC_RELATED = """
## Payments
When an order is processed, a receipt is generated immediately after the
card is charged.
"""

# A genuinely unrelated pair, to confirm similarity is meaningfully lower,
# not just uniformly high regardless of content.
CODE_UNRELATED = """
def format_log_line(level, message, timestamp):
    return f"[{timestamp}] {level.upper()}: {message}"
"""

DOC_UNRELATED = """
## Installation
Run `pip install docbadger` to install the package, then add the Action
to your workflow file.
"""


def main():
    if not os.environ.get("LLM_API_KEY"):
        print("ERROR: export LLM_API_KEY first.", file=sys.stderr)
        sys.exit(1)

    print("Requesting real embeddings from OpenRouter (text-embedding-3-small)...\n")
    texts = [CODE_RELATED, DOC_RELATED, CODE_UNRELATED, DOC_UNRELATED]
    vectors = embed_texts(texts)
    code_related_vec, doc_related_vec, code_unrelated_vec, doc_unrelated_vec = vectors

    related_similarity = cosine_similarity(code_related_vec, doc_related_vec)
    unrelated_similarity = cosine_similarity(code_related_vec, doc_unrelated_vec)
    cross_check_similarity = cosine_similarity(code_unrelated_vec, doc_related_vec)

    print(f"Similarity — process_payment code <-> Payments doc (should be HIGH):      {related_similarity:.4f}")
    print(f"Similarity — process_payment code <-> Installation doc (should be LOW):   {unrelated_similarity:.4f}")
    print(f"Similarity — logging code <-> Payments doc (should also be LOW):          {cross_check_similarity:.4f}")

    print("\n--- Sanity check ---")
    if related_similarity > unrelated_similarity and related_similarity > cross_check_similarity:
        print("PASS: the genuinely related pair scored highest, as expected.")
    else:
        print("UNEXPECTED: a related pair did not score highest — investigate before trusting this stage.")

    print(f"\n(For reference, our default threshold is 0.75 — related={related_similarity:.4f} "
          f"{'clears' if related_similarity >= 0.75 else 'does NOT clear'} it.)")


if __name__ == "__main__":
    main()
