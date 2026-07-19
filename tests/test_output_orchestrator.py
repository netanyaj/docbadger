import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corrector import CorrectionStatus, CorrectorResult
from validator import ValidationStatus, ValidatorResult
from output_orchestrator import PipelineFinding, build_orchestration_plan


def _finding(**overrides):
    defaults = dict(
        filepath="docs/auth.md",
        qualified_id="src/auth.py::login",
        heading_path="Authentication > Login",
        stale=True,
        diagnosis="login() gained a new required parameter.",
        tier="high",
    )
    defaults.update(overrides)
    return PipelineFinding(**defaults)


def test_verified_finding_produces_verified_entry():
    plan = build_orchestration_plan([_finding(stale=False, tier=None)])
    assert plan.comment_entries[0].kind == "verified"


def test_check_incomplete_finding_when_verifier_errored():
    plan = build_orchestration_plan([_finding(stale=None, tier=None, diagnosis="[LLM CALL FAILED]")])
    assert plan.comment_entries[0].kind == "check_incomplete"
    assert "LLM CALL FAILED" in plan.comment_entries[0].detail


def test_low_tier_finding_flagged_without_corrector_call():
    plan = build_orchestration_plan([_finding(tier="low")])
    assert plan.comment_entries[0].kind == "flagged_low_confidence"


def test_corrector_abstention_flagged_with_status_and_rationale():
    cr = CorrectorResult(status=CorrectionStatus.ABSTAINED_DIAGNOSIS, old_text=None, new_text=None, rationale="not enough context")
    plan = build_orchestration_plan([_finding(corrector_result=cr)])
    entry = plan.comment_entries[0]
    assert entry.kind == "flagged_abstained"
    assert "abstained_diagnosis" in entry.detail
    assert "not enough context" in entry.detail


def test_approved_correction_shown_ready_to_apply_in_comment_only():
    cr = CorrectorResult(status=CorrectionStatus.PROPOSED, old_text="old", new_text="new", rationale="fix")
    vr = ValidatorResult(status=ValidationStatus.APPROVED, old_text="old", new_text="new", rationale="checks out")
    plan = build_orchestration_plan([_finding(corrector_result=cr, validator_result=vr)])
    entry = plan.comment_entries[0]
    assert entry.kind == "correction_ready"
    assert "Replace: `old`" in entry.detail
    assert "With: `new`" in entry.detail
    # No branch/PR concept exists anywhere in this module — nothing to assert
    # its absence against beyond the fact that OrchestrationPlan has no such field.


def test_rejected_correction_surfaces_draft_but_not_marked_ready():
    cr = CorrectorResult(status=CorrectionStatus.PROPOSED, old_text="old", new_text="new", rationale="fix")
    vr = ValidatorResult(status=ValidationStatus.REJECTED_STYLE, old_text="old", new_text="new", rationale="reads awkwardly")
    plan = build_orchestration_plan([_finding(corrector_result=cr, validator_result=vr)])
    entry = plan.comment_entries[0]
    assert entry.kind == "flagged_rejected"
    assert "reads awkwardly" in entry.detail
    assert "not marked ready to apply" in entry.detail


def test_multiple_findings_across_all_kinds_in_one_plan():
    findings = [
        _finding(stale=False, tier=None),
        _finding(tier="low"),
        _finding(corrector_result=CorrectorResult(CorrectionStatus.ABSTAINED_MECHANICAL, None, None, "couldn't match text")),
        _finding(
            corrector_result=CorrectorResult(CorrectionStatus.PROPOSED, "a", "b", "fix"),
            validator_result=ValidatorResult(ValidationStatus.APPROVED, "a", "b", "good"),
        ),
    ]
    plan = build_orchestration_plan(findings)
    assert len(plan.comment_entries) == 4
    kinds = {e.kind for e in plan.comment_entries}
    assert kinds == {"verified", "flagged_low_confidence", "flagged_abstained", "correction_ready"}
