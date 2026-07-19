"""
Output Orchestrator — DocBadger pipeline, Milestone 4 final stage.

Pure planning logic only: classifies every (function, doc section) finding
from the Verifier -> Confidence Rubric -> Corrector -> Validator chain into
exactly one outcome, and produces the entries the summary comment renders.

v1 scope, confirmed explicitly: DocBadger never creates a branch or opens a
PR. Every outcome — including an APPROVED correction — is surfaced in the
same single summary comment on the PR that triggered the run, via
comment_builder.build_final_comment. An approved correction is shown as a
ready-to-apply old->new suggestion the reviewer can copy by hand; it is not
committed anywhere automatically. (An earlier version of this module did
create a companion branch/PR — that was built on a misread of an explicit
instruction to keep this comment-only, and was fully reverted. See the
Engineering Decision Log for the corrected entries and the Case Study
Journal for the honest account of the mistake.)
"""

from dataclasses import dataclass, field
from typing import Optional

from corrector import CorrectionStatus
from validator import ValidationStatus


@dataclass
class PipelineFinding:
    """Everything the Orchestrator needs to know about one (function, doc
    section) check, regardless of how far through the pipeline it got."""
    filepath: str
    qualified_id: str
    heading_path: str
    stale: Optional[bool]          # True / False / None (verifier error)
    diagnosis: str
    tier: Optional[str] = None      # "high" | "medium" | "low" | None if not stale
    corrector_result: Optional[object] = None   # corrector.CorrectorResult
    validator_result: Optional[object] = None    # validator.ValidatorResult


@dataclass
class CommentEntry:
    kind: str            # see build_orchestration_plan for the full set of kinds
    qualified_id: str
    heading_path: str
    detail: str


@dataclass
class OrchestrationPlan:
    comment_entries: list = field(default_factory=list)   # CommentEntry


def build_orchestration_plan(findings: list) -> OrchestrationPlan:
    """Pure function — no I/O. Classifies every finding into exactly one
    outcome kind for the summary comment. Nothing in this function writes
    anywhere; it only decides what text to show and how to label it.
    """
    comment_entries = []

    for f in findings:
        if f.stale is False:
            comment_entries.append(CommentEntry("verified", f.qualified_id, f.heading_path, ""))
            continue

        if f.stale is None:
            comment_entries.append(CommentEntry("check_incomplete", f.qualified_id, f.heading_path, f.diagnosis))
            continue

        # f.stale is True from here down.
        if f.tier == "low" or f.corrector_result is None:
            comment_entries.append(CommentEntry("flagged_low_confidence", f.qualified_id, f.heading_path, f.diagnosis))
            continue

        cr = f.corrector_result
        if cr.status != CorrectionStatus.PROPOSED:
            comment_entries.append(CommentEntry(
                "flagged_abstained", f.qualified_id, f.heading_path,
                f"{cr.status.value}: {cr.rationale}",
            ))
            continue

        vr = f.validator_result
        if vr is None:
            # Defensive: a correctly-wired main.py always validates a proposed
            # correction. If this ever fires, treat it as a flag, not a silent drop.
            comment_entries.append(CommentEntry(
                "flagged_abstained", f.qualified_id, f.heading_path,
                "Corrector proposed a correction but it was never validated — treating as unresolved.",
            ))
            continue

        if vr.status == ValidationStatus.APPROVED:
            comment_entries.append(CommentEntry(
                "correction_ready", f.qualified_id, f.heading_path,
                f"{vr.rationale}\n"
                f"Replace: `{vr.old_text}`\n"
                f"With: `{vr.new_text}`",
            ))
        else:
            comment_entries.append(CommentEntry(
                "flagged_rejected", f.qualified_id, f.heading_path,
                f"{vr.status.value}: {vr.rationale}\n"
                f"Drafted correction (not marked ready to apply) —\n"
                f"Replace: `{vr.old_text}`\n"
                f"With: `{vr.new_text}`",
            ))

    return OrchestrationPlan(comment_entries=comment_entries)
