"""
Regression suite against a larger, more realistic fixture set (5 code
files, 3 doc files) — distinct from the smaller, single-behavior fixtures
used elsewhere. This is the "peer review" pass: real messy-repo cases in
one place, tested together, as a standing check that survives future
milestones' changes.

Deliberately NOT the same thing as Milestone 5's eval dataset: this tests
whether the *indexing/linking pipeline* behaves correctly (objectively
checkable), not whether the *LLM's staleness judgment* agrees with a human
(a fuzzier question, answered separately).

These fixtures are static, real files (not generated in a temp dir) so
they're inspectable by a human reviewer, not just by the test runner.
No git repo is needed here — build_index(persist=False) never touches git.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import build_index, get_linked_doc_sections

FIXTURES_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "regression")


def _fake_embed_fn(texts):
    """Not testing embedding correctness here (that's embedding_linker's
    job) — just needs to exist so build_index can run without a real API
    call. Every fixture case here is designed to be resolved by the
    heuristic linker alone."""
    return [[0.0] for _ in texts]


def _build():
    return build_index(root=FIXTURES_ROOT, embed_fn=_fake_embed_fn, persist=False)


def test_login_links_to_specific_section():
    index = _build()
    sections = get_linked_doc_sections("auth.py::AuthClient.login", index)
    assert any(s.endswith("Auth > Login") for s in sections)


def test_register_links_to_specific_section():
    index = _build()
    sections = get_linked_doc_sections("auth.py::AuthClient.register", index)
    assert any(s.endswith("Auth > Registration") for s in sections)


def test_three_level_nesting_keeps_only_deepest_link():
    # logout is mentioned inside "### Session Handling", nested two levels
    # under "# Auth" — the ancestor sections ("Auth", "Auth > Login") also
    # technically contain that mention, and must NOT show up as duplicates.
    index = _build()
    sections = get_linked_doc_sections("auth.py::AuthClient.logout", index)
    assert sections == ["docs/auth.md::Auth > Login > Session Handling"]


def test_ambiguous_bare_mention_links_both_same_named_methods():
    index = _build()
    payments_sections = get_linked_doc_sections(
        "payments.py::PaymentProcessor.process_payment", index
    )
    billing_sections = get_linked_doc_sections(
        "billing.py::BillingService.process_payment", index
    )
    assert payments_sections, "PaymentProcessor.process_payment should be linked"
    assert billing_sections, "BillingService.process_payment should be linked"
    assert payments_sections == billing_sections, (
        "a bare 'process_payment' mention should link to both classes' "
        "methods equally — heuristic linking doesn't disambiguate by class"
    )


def test_refund_links_correctly_and_is_not_confused_with_payments():
    index = _build()
    sections = get_linked_doc_sections("payments.py::PaymentProcessor.refund", index)
    assert sections == ["docs/billing.md::Billing > Refunds"]


def test_undocumented_internal_function_has_no_link():
    index = _build()
    sections = get_linked_doc_sections("notifications.py::_internal_helper", index)
    assert sections == []


def test_documented_function_links_correctly():
    index = _build()
    sections = get_linked_doc_sections("notifications.py::send_notification", index)
    assert sections == ["docs/misc.md::Misc > Notifications"]


def test_stale_doc_reference_produces_no_link_not_a_crash():
    # misc.md references `load_configuration()`, which doesn't exist
    # anywhere in the codebase — the real function is load_settings.
    # This must resolve to "no link," never an exception.
    index = _build()
    sections = get_linked_doc_sections("config.py::load_settings", index)
    assert sections == []


def test_no_chunk_or_section_is_silently_dropped_from_indexing():
    # Sanity check on the parsers themselves, run against this larger set:
    # every file we wrote should actually produce chunks/sections.
    index = _build()
    assert len(index["code_chunks"]) >= 12  # 5 files' worth of functions/classes/methods
    assert len(index["doc_sections"]) >= 7   # 3 files' worth of headings
