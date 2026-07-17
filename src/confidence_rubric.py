"""
Confidence Rubric — deterministic, explainable scoring for how much trust
a proposed correction deserves, per Engineering Decision Log Entry 2: a
designed rubric applied to structured facts, never an LLM self-reported
number.

Four factors, each contributing points toward a 0-100 score:
  1. Change type    — signature change (higher) vs. body-only (lower).
  2. Link certainty  — how the doc section was found: exact heuristic match
                        (highest), bare leaf match (medium), embedding
                        similarity (lowest). Per the Milestone 4 kickoff
                        decision: heuristic ALWAYS outranks embedding, full
                        stop — no embedding score, however high, currently
                        overrides a heuristic match. (Revisit in Milestone 5
                        once real eval data exists — see Future Product
                        Opportunities note on ambiguous-heuristic vs.
                        clean-embedding trade-offs.)
  3. Blast radius     — how many total doc sections this one code change is
                        linked to. Fewer = more confident this is *the*
                        affected section; more = diffuse, ambiguous risk.
  4. Historical accuracy — per-repo, starts neutral. Real feedback-driven
                        scoring doesn't exist until Milestone 6 (US-5); this
                        factor is a fixed placeholder for now, kept in the
                        formula's shape so wiring in real feedback later
                        doesn't require redesigning the rubric.

All point values and tier boundaries below are PROVISIONAL — explicitly
placeholder numbers, not yet validated against real data. Milestone 5's
hand-labeled eval dataset is what actually calibrates these; until then,
treat this as "a reasonable, explainable first attempt," not "the right
answer." Same honesty standard as the embedding similarity threshold.
"""

from dataclasses import dataclass, field

# --- Point values (provisional — see module docstring) ---
CHANGE_TYPE_POINTS = {"signature": 40, "body_only": 15}
LINK_CERTAINTY_POINTS = {"exact": 30, "leaf": 20, "embedding": 10}
NEUTRAL_HISTORICAL_ACCURACY_POINTS = 10  # fixed until Milestone 6 feedback loop exists

# --- Tier boundaries (provisional — see module docstring) ---
HIGH_THRESHOLD = 70
MEDIUM_THRESHOLD = 40


def _blast_radius_points(blast_radius: int) -> int:
    if blast_radius <= 1:
        return 20
    if blast_radius == 2:
        return 10
    return 0


def _tier_for_score(score: int) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _cap_tier_for_embedding(tier: str, link_source: str) -> str:
    """Enforces the Milestone 4 decision as an actual guarantee, not just a
    likely outcome of point arithmetic: a simple additive score does NOT
    guarantee heuristic-sourced links always outscore embedding-sourced
    ones (a favorable-change-type, tight-blast-radius embedding match can
    out-point an unfavorable heuristic one). Rather than fragile point
    tuning, the tier itself is capped directly: an embedding-sourced link
    can never reach "high," regardless of how favorable its other factors
    are. The raw score is still computed transparently for the breakdown."""
    if link_source == "embedding" and tier == "high":
        return "medium"
    return tier


@dataclass
class ConfidenceResult:
    score: int
    tier: str  # "high" | "medium" | "low"
    breakdown: dict = field(default_factory=dict)  # factor name -> points, for transparency


def score_confidence(
    change_type: str,
    link_source: str,
    blast_radius: int,
    historical_accuracy_points: int = NEUTRAL_HISTORICAL_ACCURACY_POINTS,
) -> ConfidenceResult:
    """Pure scoring function — no I/O, fully deterministic, fully testable
    without touching the index or the pipeline at all."""
    change_points = CHANGE_TYPE_POINTS.get(change_type, CHANGE_TYPE_POINTS["body_only"])
    certainty_points = LINK_CERTAINTY_POINTS.get(link_source, LINK_CERTAINTY_POINTS["embedding"])
    radius_points = _blast_radius_points(blast_radius)

    breakdown = {
        "change_type": change_points,
        "link_certainty": certainty_points,
        "blast_radius": radius_points,
        "historical_accuracy": historical_accuracy_points,
    }
    score = sum(breakdown.values())
    tier = _cap_tier_for_embedding(_tier_for_score(score), link_source)
    return ConfidenceResult(score=score, tier=tier, breakdown=breakdown)


def score_confidence_for_link(modified_function, section_id: str, index: dict) -> ConfidenceResult:
    """Convenience wrapper: pulls link_source and blast_radius directly
    from a build_index() result, and change_type from the ModifiedFunction
    itself (from diff_analyzer.get_modified_functions) — so callers don't
    need to manually assemble the raw inputs.

    modified_function: a diff_analyzer.ModifiedFunction, which carries its
    own qualified_id and change_type — passed as-is rather than looked up,
    since change_type isn't something the index stores (only diff_analyzer
    computes it, at diff time).
    """
    from indexer import get_link_sources, get_linked_doc_sections

    sources = get_link_sources(modified_function.qualified_id, index)
    link_source = sources.get(section_id, "embedding")
    blast_radius = len(get_linked_doc_sections(modified_function.qualified_id, index))
    return score_confidence(
        change_type=modified_function.change_type,
        link_source=link_source,
        blast_radius=blast_radius,
    )
