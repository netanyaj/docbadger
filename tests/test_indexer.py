import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import build_index, get_linked_doc_sections, get_link_sources

CODE_FILE = '''def process_payment(order):
    if order.total > FRAUD_THRESHOLD:
        flag_for_review(order)
        return None
    charge_card(order.card, order.total)
    return Receipt(order)


def send_email(to, subject, body):
    """Send an email."""
    smtp_client.send(to, subject, body)
'''

DOC_FILE = """# Docs

## Payments
When an order is processed, a receipt is generated immediately after the
card is charged.

## Emails
Call `send_email(to, subject, body)` to send an email.
"""


def _run(repo_dir, *args):
    return subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True,
    )


def _build_repo_with_fixtures_and_fake_origin():
    bare_dir = tempfile.mkdtemp()
    _run(bare_dir, "init", "--bare", "-q")

    work_dir = tempfile.mkdtemp()
    _run(work_dir, "init", "-q")
    _run(work_dir, "config", "user.email", "test@example.com")
    _run(work_dir, "config", "user.name", "Test Runner")
    _run(work_dir, "remote", "add", "origin", bare_dir)

    os.makedirs(os.path.join(work_dir, "docs"))
    with open(os.path.join(work_dir, "payments.py"), "w") as f:
        f.write(CODE_FILE)
    with open(os.path.join(work_dir, "docs", "guide.md"), "w") as f:
        f.write(DOC_FILE)

    _run(work_dir, "add", ".")
    _run(work_dir, "commit", "-q", "-m", "initial commit")
    _run(work_dir, "push", "origin", "HEAD:refs/heads/main")

    return work_dir


def _fake_embed_fn(texts):
    """Assigns 'payments'-topic vectors to anything mentioning payment/order/
    receipt/card, and a distinct vector to everything else — enough to
    exercise the embedding fallback deterministically without a real API."""
    vectors = []
    for text in texts:
        lowered = text.lower()
        if any(kw in lowered for kw in ("payment", "order", "receipt", "card")):
            vectors.append([1.0, 0.0])
        else:
            vectors.append([0.0, 1.0])
    return vectors


def test_docs_root_restricts_doc_scanning_but_not_code_scanning():
    """DOCS_PATH (action.yml `docs_path` input) is meant to restrict DOC
    scanning to a subdirectory, per Architecture Section 4 ("Restrict doc
    scanning to a docs directory"). Code scanning must NOT be restricted
    by it -- a code change relevant to a doc section can live anywhere in
    the repo, not just under docs/. Reproduces the real bug directly: a
    second doc file placed OUTSIDE docs_root must be invisible to the
    index once docs_root is passed, while payments.py (at the repo root,
    outside docs_root) must still be parsed and linkable."""
    work_dir = _build_repo_with_fixtures_and_fake_origin()

    # A second, real doc file living OUTSIDE the docs/ subdirectory --
    # should never be scanned once docs_root=docs/ is passed.
    with open(os.path.join(work_dir, "ROOT_NOTES.md"), "w") as f:
        f.write("# Root Notes\n\n## Stray\nThis should not be scanned once docs_root is set.\n")

    docs_root = os.path.join(work_dir, "docs")
    index = build_index(root=work_dir, docs_root=docs_root, embed_fn=_fake_embed_fn, persist=False)

    doc_section_ids = list(index["doc_sections"].keys())
    assert not any("ROOT_NOTES.md" in sid for sid in doc_section_ids), (
        "docs_root did not restrict doc scanning -- a doc file outside "
        "docs_root was still parsed and indexed"
    )
    assert any("guide.md" in sid for sid in doc_section_ids), (
        "docs_root over-restricted -- the real doc file inside docs_root "
        "was not found"
    )

    # Code scanning must be unaffected -- payments.py lives at the repo
    # root, outside docs_root, and must still be parsed and linkable.
    sections = get_linked_doc_sections("payments.py::send_email", index)
    assert any("Emails" in s for s in sections), (
        "code scanning was incorrectly restricted by docs_root -- "
        "payments.py::send_email should still link normally"
    )


def test_heuristic_link_found_via_backtick_mention():
    work_dir = _build_repo_with_fixtures_and_fake_origin()
    index = build_index(root=work_dir, embed_fn=_fake_embed_fn, persist=False)
    sections = get_linked_doc_sections("payments.py::send_email", index)
    assert any("Emails" in s for s in sections)


def test_ancestor_section_duplicates_are_dropped():
    # DOC_FILE has a top-level "# Docs" heading wrapping both "## Payments"
    # and "## Emails" — the same shape that caused the real duplicate-flag
    # bug (a parent section's text contains its child's mentions too).
    work_dir = _build_repo_with_fixtures_and_fake_origin()
    index = build_index(root=work_dir, embed_fn=_fake_embed_fn, persist=False)
    sections = get_linked_doc_sections("payments.py::send_email", index)

    # Should link ONLY to the specific "Docs > Emails" section, not also
    # to the bare, all-containing "Docs" top-level section.
    assert not any(s.endswith("::Docs") for s in sections)
    assert any(s.endswith("Docs > Emails") for s in sections)


def test_embedding_fallback_links_unnamed_behavior_match():
    work_dir = _build_repo_with_fixtures_and_fake_origin()
    index = build_index(root=work_dir, embed_fn=_fake_embed_fn, persist=False)
    # process_payment is never named in the doc — only the embedding
    # fallback (with our fake topic-based vectors) can find this link.
    sections = get_linked_doc_sections("payments.py::process_payment", index)
    assert any("Payments" in s for s in sections)


def test_persist_true_writes_cache_and_pushes_to_backstop_branch():
    work_dir = _build_repo_with_fixtures_and_fake_origin()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        build_index(root=work_dir, embed_fn=_fake_embed_fn, persist=True)
        log = _run(work_dir, "log", "origin/docbadger/index", "--oneline").stdout.strip()
    finally:
        os.chdir(original_cwd)

    assert len(log.splitlines()) == 1  # one commit pushed to the backstop branch


def test_cache_garbage_collects_orphaned_entries_when_content_changes():
    import json

    work_dir = _build_repo_with_fixtures_and_fake_origin()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        build_index(root=work_dir, embed_fn=_fake_embed_fn, persist=True)
        cache_path = os.path.join(work_dir, ".docbadger_cache", "embeddings.json")
        with open(cache_path) as f:
            first_cache = json.load(f)

        # Change process_payment's code — it has no heuristic doc match, so
        # it's the chunk that actually goes through embedding and gets
        # cached. (send_email, by contrast, is resolved via heuristic
        # matching and is never embedded/cached at all — modifying it
        # wouldn't create anything to orphan.)
        with open(os.path.join(work_dir, "payments.py")) as f:
            content = f.read()
        with open(os.path.join(work_dir, "payments.py"), "w") as f:
            f.write(content.replace("order.total > FRAUD_THRESHOLD", "order.total > NEW_THRESHOLD"))

        build_index(root=work_dir, embed_fn=_fake_embed_fn, persist=True)
        with open(cache_path) as f:
            second_cache = json.load(f)
    finally:
        os.chdir(original_cwd)

    orphaned = set(first_cache) - set(second_cache)
    assert len(orphaned) > 0, "expected the old content hash to be garbage collected"


def test_second_run_reuses_cache_and_skips_unchanged_embeddings():
    work_dir = _build_repo_with_fixtures_and_fake_origin()
    call_counts = {"n": 0}

    def counting_embed_fn(texts):
        call_counts["n"] += len(texts)
        return _fake_embed_fn(texts)

    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        build_index(root=work_dir, embed_fn=counting_embed_fn, persist=True)
        first_run_calls = call_counts["n"]

        call_counts["n"] = 0
        build_index(root=work_dir, embed_fn=counting_embed_fn, persist=True)
        second_run_calls = call_counts["n"]
    finally:
        os.chdir(original_cwd)

    assert first_run_calls > 0
    assert second_run_calls == 0  # nothing changed — cache should cover everything


def test_cache_hit_miss_counts_reported_for_run_logging():
    # Architecture Section 15's structured run log needs real embedding
    # cache hit/miss numbers, not just "did it work." First run against a
    # fresh cache: whatever is embedding-linked must be a miss (nothing to
    # hit yet). Second run, nothing changed: the same lookups must now be
    # hits, and there must be zero misses -- same fixture/pattern as
    # test_second_run_reuses_cache_and_skips_unchanged_embeddings, just
    # asserting the counts build_index now returns instead of a wrapped
    # embed_fn's call count.
    work_dir = _build_repo_with_fixtures_and_fake_origin()
    original_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        first_index = build_index(root=work_dir, embed_fn=_fake_embed_fn, persist=True)
        second_index = build_index(root=work_dir, embed_fn=_fake_embed_fn, persist=True)
    finally:
        os.chdir(original_cwd)

    assert first_index["cache_misses"] > 0
    assert first_index["cache_hits"] == 0

    assert second_index["cache_hits"] > 0
    assert second_index["cache_misses"] == 0
    assert second_index["cache_hits"] == first_index["cache_misses"]


def test_link_sources_are_labeled_correctly():
    work_dir = _build_repo_with_fixtures_and_fake_origin()
    index = build_index(root=work_dir, embed_fn=_fake_embed_fn, persist=False)

    # send_email is heuristically matched (doc backtick-references it directly).
    email_sources = get_link_sources("payments.py::send_email", index)
    assert set(email_sources.values()) == {"exact"}

    # process_payment is never named in the doc — only reachable via the
    # embedding fallback (see test_embedding_fallback_links_unnamed_behavior_match).
    payment_sources = get_link_sources("payments.py::process_payment", index)
    assert set(payment_sources.values()) == {"embedding"}
