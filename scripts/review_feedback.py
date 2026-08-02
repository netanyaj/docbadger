#!/usr/bin/env python3
"""
Feedback batch-review report — Milestone 6, Thread 1.

Entry 45 committed to feedback being batched and human-reviewed before it
ever affects the confidence rubric — but nothing existed yet to make that
review actually practical. This script closes that gap: it pulls
feedback.json from the docbadger/index branch and produces a readable
summary a human can act on, without inventing any automatic rubric-tuning
logic (that stays deliberately out of scope, per Entry 45 and Future
Product Opportunities #13 — this script informs a human decision, it
doesn't make one).

Usage:
    python scripts/review_feedback.py
"""

import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from index_branch_sync import pull_index

FEEDBACK_FILENAME = "feedback.json"
MIN_RECORDS_FOR_PATTERNS = 5  # don't call out a "pattern" from too few records — same
                               # "don't trust small-n" discipline as the eval dataset


def load_feedback() -> dict:
    return pull_index(filename=FEEDBACK_FILENAME)


def build_report(store: dict) -> str:
    records = list(store.values())
    lines = ["# DocBadger Feedback Review", ""]

    if not records:
        lines.append("_No feedback recorded yet._")
        return "\n".join(lines)

    lines.append(f"Total feedback records: **{len(records)}**")
    lines.append("")

    verdict_counts = Counter(r.get("verdict") for r in records)
    lines.append("## By verdict")
    for verdict in ("accepted", "rejected", "unsure"):
        lines.append(f"- {verdict}: {verdict_counts.get(verdict, 0)}")
    lines.append("")

    by_kind = defaultdict(Counter)
    for r in records:
        by_kind[r.get("kind", "unknown")][r.get("verdict")] += 1
    lines.append("## By finding kind")
    for kind, counts in sorted(by_kind.items()):
        total = sum(counts.values())
        breakdown = ", ".join(f"{v}={c}" for v, c in counts.items())
        lines.append(f"- **{kind}** ({total} total): {breakdown}")
    lines.append("")

    by_tier = defaultdict(Counter)
    for r in records:
        tier = r.get("tier") or "n/a"
        by_tier[tier][r.get("verdict")] += 1
    lines.append("## By confidence tier")
    for tier, counts in sorted(by_tier.items()):
        total = sum(counts.values())
        breakdown = ", ".join(f"{v}={c}" for v, c in counts.items())
        lines.append(f"- **{tier}** ({total} total): {breakdown}")
    lines.append("")

    with_reason = [r for r in records if r.get("reason_context")]
    if with_reason:
        lines.append("## Reviewer notes")
        for r in with_reason:
            lines.append(f"- [{r.get('verdict')}] `{r.get('qualified_id')}` — {r.get('reason_context')}")
        lines.append("")

    if len(records) < MIN_RECORDS_FOR_PATTERNS:
        lines.append(
            f"_Only {len(records)} record(s) so far — not enough to draw conclusions about "
            "rubric adjustments yet. This report gets more useful as more real PRs run._"
        )
    else:
        rejected_ready = by_kind.get("correction_ready", Counter()).get("rejected", 0)
        if rejected_ready:
            lines.append(
                f"⚠️ {rejected_ready} `correction_ready` finding(s) were marked rejected by a "
                "reviewer despite passing Validator — worth investigating whether Validator's "
                "accuracy/style checks need tuning (see Future Product Opportunities #12)."
            )

    return "\n".join(lines)


def main():
    store = load_feedback()
    report = build_report(store)
    print(report)


if __name__ == "__main__":
    main()
