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
