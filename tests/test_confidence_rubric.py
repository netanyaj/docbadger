import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from confidence_rubric import score_confidence, score_confidence_for_link, HIGH_THRESHOLD, MEDIUM_THRESHOLD


def test_signature_change_scores_higher_than_body_only():
    signature_result = score_confidence(change_type="signature", link_source="exact", blast_radius=1)
    body_only_result = score_confidence(change_type="body_only", link_source="exact", blast_radius=1)
    assert signature_result.score > body_only_result.score


def test_exact_link_scores_higher_than_leaf():
    exact_result = score_confidence(change_type="signature", link_source="exact", blast_radius=1)
    leaf_result = score_confidence(change_type="signature", link_source="leaf", blast_radius=1)
    assert exact_result.score > leaf_result.score


def test_leaf_link_scores_higher_than_embedding():
    leaf_result = score_confidence(change_type="signature", link_source="leaf", blast_radius=1)
    embedding_result = score_confidence(change_type="signature", link_source="embedding", blast_radius=1)
    assert leaf_result.score > embedding_result.score


def test_embedding_link_can_never_reach_high_tier_even_with_best_other_factors():
    # The raw additive score does NOT by itself guarantee this (a favorable
    # embedding case can out-point an unfavorable heuristic one — see
    # confidence_rubric._cap_tier_for_embedding's docstring for the honest
    # arithmetic). The tier cap is what actually enforces the Milestone 4
    # decision: embedding-sourced links never reach "high," full stop.
    best_case_embedding = score_confidence(change_type="signature", link_source="embedding", blast_radius=1)
    assert best_case_embedding.tier != "high"
    assert best_case_embedding.tier == "medium"


def test_heuristic_link_can_reach_high_tier():
    result = score_confidence(change_type="signature", link_source="exact", blast_radius=1)
    assert result.tier == "high"


def test_smaller_blast_radius_scores_higher():
    single_link = score_confidence(change_type="signature", link_source="exact", blast_radius=1)
    two_links = score_confidence(change_type="signature", link_source="exact", blast_radius=2)
    many_links = score_confidence(change_type="signature", link_source="exact", blast_radius=5)
    assert single_link.score > two_links.score > many_links.score


def test_best_case_reaches_high_tier():
    result = score_confidence(change_type="signature", link_source="exact", blast_radius=1)
    assert result.tier == "high"
    assert result.score >= HIGH_THRESHOLD


def test_worst_case_reaches_low_tier():
    result = score_confidence(change_type="body_only", link_source="embedding", blast_radius=5)
    assert result.tier == "low"
    assert result.score < MEDIUM_THRESHOLD


def test_breakdown_is_transparent_and_sums_to_score():
    result = score_confidence(change_type="signature", link_source="leaf", blast_radius=2)
    assert sum(result.breakdown.values()) == result.score
    assert set(result.breakdown.keys()) == {
        "change_type", "link_certainty", "blast_radius", "historical_accuracy"
    }


def test_unknown_change_type_defaults_conservatively():
    # Should never crash on an unexpected value — falls back to the lower
    # (body_only) point value rather than raising or over-crediting.
    result = score_confidence(change_type="something_unexpected", link_source="exact", blast_radius=1)
    body_only_result = score_confidence(change_type="body_only", link_source="exact", blast_radius=1)
    assert result.score == body_only_result.score


class _FakeModifiedFunction:
    def __init__(self, qualified_id, change_type):
        self.qualified_id = qualified_id
        self.change_type = change_type


def test_score_confidence_for_link_pulls_change_type_from_modified_function():
    fake_index = {
        "links": {
            "auth.py::login": {"docs/auth.md::Login": "exact"},
        }
    }
    fn = _FakeModifiedFunction("auth.py::login", change_type="signature")
    result = score_confidence_for_link(fn, "docs/auth.md::Login", fake_index)
    assert result.breakdown["change_type"] == score_confidence(
        change_type="signature", link_source="exact", blast_radius=1
    ).breakdown["change_type"]
    assert result.tier == "high"


def test_score_confidence_for_link_computes_blast_radius_from_index():
    fake_index = {
        "links": {
            "auth.py::login": {
                "docs/auth.md::Login": "exact",
                "docs/auth.md::Quickstart": "leaf",
            },
        }
    }
    fn = _FakeModifiedFunction("auth.py::login", change_type="signature")
    result = score_confidence_for_link(fn, "docs/auth.md::Login", fake_index)
    # blast_radius=2 here, not 1 — should score lower than the single-link case above.
    single_link_result = score_confidence(change_type="signature", link_source="exact", blast_radius=1)
    assert result.score < single_link_result.score
