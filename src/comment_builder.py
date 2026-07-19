"""
Comment Builder — pure function that turns pipeline results into the PR
summary comment text (US-6 format). Separated from main.py specifically so
it can be unit tested without needing a real GitHub token or PR.
"""


def build_comment(meaningful_count: int, checked_results: list, error_count: int) -> str:
    lines = [
        "## DocBadger Doc Check",
        "",
        f"- Meaningful code changes detected: **{meaningful_count}**",
        f"- Changes with a known doc link checked: **{len(checked_results)}**",
        f"- Sections flagged as stale: **{sum(1 for _, _, v in checked_results if v['stale'] is True)}**",
    ]
    if error_count:
        lines.append(f"- Checks that could not complete: **{error_count}** (see logs)")
    lines.append("")

    if not checked_results:
        lines.append(
            "_No changed code was found to have a linked documentation section "
            "this run — either nothing meaningful changed, or no doc section "
            "currently references the changed code (by name or by meaning)._"
        )
    else:
        for fn, heading, verdict in checked_results:
            if verdict["stale"] is True:
                lines.append(f"### ⚠️ Possibly stale: `{fn.qualified_id}` → \"{heading}\"")
                lines.append(f"{verdict['diagnosis']}")
            elif verdict["stale"] is False:
                lines.append(f"### ✅ Verified accurate: `{fn.qualified_id}` → \"{heading}\"")
            else:
                lines.append(f"### ⚠️ Check incomplete: `{fn.qualified_id}` → \"{heading}\"")
                lines.append(f"{verdict['diagnosis']}")
            lines.append("")

    return "\n".join(lines)


_KIND_HEADERS = {
    "verified": "✅ Verified accurate",
    "check_incomplete": "⚠️ Check incomplete",
    "flagged_low_confidence": "⚠️ Possibly stale (low confidence — not auto-drafted)",
    "flagged_abstained": "⚠️ Possibly stale (correction could not be drafted)",
    "flagged_rejected": "⚠️ Possibly stale (draft did not pass validation)",
    "correction_ready": "🛠️ Correction ready to apply",
}


def build_final_comment(meaningful_count: int, comment_entries: list, error_count: int) -> str:
    """Milestone 4's complete summary comment, built from an
    output_orchestrator.OrchestrationPlan's comment_entries — supersedes
    build_comment() above as of the Output Orchestrator stage. build_comment
    is kept as-is (and still tested) since it reflects the Milestone 2
    flag-only shape and nothing currently requires removing it.

    v1 scope, confirmed explicitly with the user: this is the ONLY place
    pipeline output is ever surfaced. No branch or PR is ever created —
    even an approved correction is shown here as a ready-to-apply, labeled
    old→new suggestion, not pushed anywhere.
    """
    stale_kinds = {"flagged_low_confidence", "flagged_abstained", "flagged_rejected", "correction_ready"}
    stale_count = sum(1 for e in comment_entries if e.kind in stale_kinds)
    ready_count = sum(1 for e in comment_entries if e.kind == "correction_ready")

    lines = [
        "## DocBadger Doc Check",
        "",
        f"- Meaningful code changes detected: **{meaningful_count}**",
        f"- Changes with a known doc link checked: **{len(comment_entries)}**",
        f"- Sections flagged as stale: **{stale_count}**",
        f"- Corrections ready to apply: **{ready_count}**",
    ]
    if error_count:
        lines.append(f"- Checks that could not complete: **{error_count}** (see logs)")
    lines.append("")

    if not comment_entries:
        lines.append(
            "_No changed code was found to have a linked documentation section "
            "this run — either nothing meaningful changed, or no doc section "
            "currently references the changed code (by name or by meaning)._"
        )
    else:
        for entry in comment_entries:
            if entry.kind == "verified":
                continue  # keep the comment focused on things that need attention
            header = _KIND_HEADERS.get(entry.kind, entry.kind)
            lines.append(f"### {header}: `{entry.qualified_id}` → \"{entry.heading_path}\"")
            if entry.detail:
                lines.append(entry.detail)
            lines.append("")

    return "\n".join(lines)
