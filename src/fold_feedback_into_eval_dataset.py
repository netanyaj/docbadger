#!/usr/bin/env python3
"""
Fold real-usage feedback into the eval dataset -- Architecture Section 19
point 5 / PRD US-5: "A defined mechanism exists ... to mark false
positives. Feedback is persisted and becomes part of the eval/regression
dataset." "false-positive markers captured from real usage are
periodically folded into the eval dataset as new labeled examples."

Scope, deliberately narrow: only feedback with verdict == "rejected" is
folded -- US-5's acceptance criteria is specifically "mark a flagged
section as 'not actually stale'," i.e. the Verifier's stale=True judgment
itself was wrong. "accepted" and "unsure" verdicts aren't folded here (no
spec language asks for it, and "accepted" isn't a new labeled example --
it just confirms the existing one was right).

Real data constraint found while building this: FeedbackSnapshot (see
feedback.py) never carried the actual (old_code, new_code) the pipeline
evaluated -- only qualified_id, heading_path, diagnosis, and the
CORRECTION's old_text/new_text (a doc-text span, not a code diff). Folding
a feedback record into a real eval case therefore requires re-fetching the
original PR's base/head SHAs (via the GitHub API) and re-deriving old_code/
new_code the exact same way the original run did, via
diff_analyzer.get_modified_functions() -- not approximating or guessing at
it. The doc section's file path was ALSO missing from the snapshot
entirely (only heading_path was captured, and heading paths aren't
globally unique across a doc set) -- fixed as part of this same build pass
by adding a `filepath` field to CommentEntry/FeedbackSnapshot/
build_feedback_block (Engineering Decision Log Entry 88). Records captured
BEFORE that fix still lack filepath; this tool falls back to a heading-
path-only search for those and refuses to guess if it's ambiguous.

This script's real-API/git-dependent parts (fetching a PR's SHAs, reading
the doc tree at a specific commit) cannot be exercised end-to-end in this
build environment (no live GitHub API access here, same limitation
disclosed in fix 8's baseline and Milestone 7) -- every pure decision
(which records are eligible, how a doc section is resolved, what the
resulting case looks like, what filename it gets) is factored into
separately unit-tested functions below, so the only untested surface is
thin I/O glue, not logic.

Usage (in a real environment, with GITHUB_TOKEN set and a checkout at each
PR's real head SHA reachable):
    python scripts/fold_feedback_into_eval_dataset.py
    python scripts/fold_feedback_into_eval_dataset.py --feedback-file local_feedback.json --dry-run
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "eval", "dataset")
FEEDBACK_FILENAME = "feedback.json"


def select_false_positive_records(feedback_store: dict) -> list:
    """feedback_store: {finding_id: record_dict, ...} (feedback.json's real
    shape). Returns only verdict == "rejected" records, each guaranteed to
    carry its own finding_id (some real records may only have it implicitly
    as the dict key, not repeated inside the value)."""
    records = []
    for finding_id, record in feedback_store.items():
        if record.get("verdict") != "rejected":
            continue
        record = dict(record)
        record.setdefault("finding_id", finding_id)
        records.append(record)
    return records


def already_folded_finding_ids(dataset_dir: str = DATASET_DIR) -> set:
    """Scans existing eval case files for source.finding_id -- a case
    already folded from a given piece of feedback must never be folded
    again on a re-run."""
    folded = set()
    for path in glob.glob(os.path.join(dataset_dir, "*.json")):
        with open(path) as f:
            case = json.load(f)
        finding_id = case.get("source", {}).get("finding_id")
        if finding_id:
            folded.add(finding_id)
    return folded


def resolve_doc_section_id(heading_path: str, filepath, doc_sections: dict):
    """doc_sections: {section_id: DocSection} from doc_parser.get_all_doc_
    sections(), where section_id is "filepath::heading_path" (doc_parser's
    own convention). Returns the matching section_id, or None if it can't
    be resolved unambiguously -- never guesses.

    If filepath is known (records captured after Entry 88's fix), this is
    a direct, certain lookup. If not (older records), falls back to
    scanning every section for a heading_path match -- safe only if
    exactly one exists across the whole doc set; refuses (returns None) on
    zero or multiple matches rather than picking one arbitrarily.
    """
    if filepath:
        candidate = f"{filepath}::{heading_path}"
        return candidate if candidate in doc_sections else None

    matches = [sid for sid, section in doc_sections.items() if section.heading_path == heading_path]
    if len(matches) == 1:
        return matches[0]
    return None  # zero or ambiguous -- refuse rather than guess


def build_eval_case_from_feedback(record: dict, old_code: str, new_code: str, doc_section_text: str) -> dict:
    """Builds a case matching the real eval/dataset/*.json schema. verdict
    == "rejected" means the human said this was NOT actually stale (US-5's
    "mark a flagged section as not actually stale") -- so the new gold
    label is verifier_expected_stale: False, a true-negative example, with
    no expected correction (nothing should have been proposed).
    """
    finding_id = record["finding_id"]
    slug = re.sub(r"[^a-z0-9]+", "_", record.get("qualified_id", "case").lower()).strip("_")[:40]
    return {
        "id": f"real_usage_{slug}_{finding_id[:8]}",
        "source": {
            "type": "real_usage_feedback",
            "repo": record.get("repo_full_name"),
            "pr_number": record.get("pr_number"),
            "finding_id": finding_id,
            "reviewer_username": record.get("reviewer_username"),
            "original_kind": record.get("kind"),
            "original_tier": record.get("tier"),
            "original_diagnosis": record.get("diagnosis"),
            "rationale": (
                "Folded from real-usage feedback (Architecture Section 19 point 5 / "
                "PRD US-5) -- a reviewer marked this DocBadger finding as a false "
                "positive (verdict=rejected). old_code/new_code were re-derived from "
                "the real PR's base/head commits via diff_analyzer.get_modified_"
                "functions, the same code path the original run used, not "
                "reconstructed or approximated."
            ),
        },
        "old_code": old_code,
        "new_code": new_code,
        "stale_doc_section": doc_section_text,
        "labels": {
            "verifier_expected_stale": False,
            "expected_pipeline_outcome": {"corrector_status": None, "validator_status": None},
            "notes_for_labeler": (
                f"Real-usage false positive, not synthetic. Original pipeline run flagged this "
                f"as kind={record.get('kind')!r} (tier={record.get('tier')!r}); reviewer "
                f"{record.get('reviewer_username')!r} marked it Rejected"
                + (f" with reason: {record.get('reason_context')!r}" if record.get("reason_context") else " with no stated reason")
                + ". Folded automatically by scripts/fold_feedback_into_eval_dataset.py -- "
                "review before treating as a fully trusted gold label, same as any new case "
                "addition (a human disagreeing with the tool is real signal, but a single "
                "disagreement isn't automatically ground truth either)."
            ),
        },
    }


def next_case_filename(case_id: str, dataset_dir: str = DATASET_DIR) -> str:
    """Matches the existing NNN_description.json numbering convention --
    continues from the highest number already present rather than
    restarting or colliding with it."""
    existing_numbers = []
    for path in glob.glob(os.path.join(dataset_dir, "*.json")):
        m = re.match(r"^(\d+)_", os.path.basename(path))
        if m:
            existing_numbers.append(int(m.group(1)))
    next_number = (max(existing_numbers) + 1) if existing_numbers else 1
    return f"{next_number:03d}_{case_id}.json"


def fold_one_record(record: dict, repo_root: str, github_repo) -> dict:
    """The real I/O path -- fetches the PR's SHAs (real GitHub API call via
    `github_repo`, a PyGithub Repository object), re-derives old_code/
    new_code via diff_analyzer against those SHAs, resolves the doc section
    via doc_parser against the CURRENT checkout (must already be at the
    PR's head SHA -- a precondition of this function, not handled inside
    it), and returns a built case dict via build_eval_case_from_feedback,
    or raises ValueError with a clear reason if any step can't be resolved
    (never silently skips without saying why).
    """
    from diff_analyzer import get_modified_functions
    from doc_parser import get_all_doc_sections

    pr = github_repo.get_pull(record["pr_number"])
    base_sha, head_sha = pr.base.sha, pr.head.sha

    modified = get_modified_functions(base_sha, head_sha)
    match = next((fn for fn in modified if fn.qualified_id == record["qualified_id"]), None)
    if match is None:
        raise ValueError(
            f"Could not find qualified_id={record['qualified_id']!r} among modified "
            f"functions between {base_sha}..{head_sha} for PR #{record['pr_number']}."
        )

    doc_sections = get_all_doc_sections(repo_root)
    section_id = resolve_doc_section_id(record.get("heading_path"), record.get("filepath"), doc_sections)
    if section_id is None:
        raise ValueError(
            f"Could not unambiguously resolve doc section for heading_path="
            f"{record.get('heading_path')!r} filepath={record.get('filepath')!r} "
            f"(finding_id={record['finding_id']})."
        )

    return build_eval_case_from_feedback(
        record, old_code=match.old_code, new_code=match.new_code,
        doc_section_text=doc_sections[section_id].text,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-file", help="local feedback.json path (default: pull from the real index branch)")
    parser.add_argument("--repo-root", default=".", help="checkout root for doc_parser -- must already be at each PR's head SHA")
    parser.add_argument("--dry-run", action="store_true", help="report what would be folded without writing any files")
    args = parser.parse_args()

    if args.feedback_file:
        with open(args.feedback_file) as f:
            feedback_store = json.load(f)
    else:
        from index_branch_sync import pull_index
        feedback_store = pull_index(filename=FEEDBACK_FILENAME)

    candidates = select_false_positive_records(feedback_store)
    already_folded = already_folded_finding_ids()
    eligible = [r for r in candidates if r["finding_id"] not in already_folded]

    print(f"{len(candidates)} rejected feedback record(s) total, "
          f"{len(candidates) - len(eligible)} already folded, {len(eligible)} to process.")

    if not eligible:
        return

    from github import Github, Auth

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not set -- required to fetch PR base/head SHAs.", file=sys.stderr)
        sys.exit(1)
    gh = Github(auth=Auth.Token(token))

    folded, failed = 0, 0
    for record in eligible:
        repo_full_name = record.get("repo_full_name")
        try:
            github_repo = gh.get_repo(repo_full_name)
            case = fold_one_record(record, args.repo_root, github_repo)
        except Exception as e:
            failed += 1
            print(f"  SKIPPED finding_id={record['finding_id']}: {e}", file=sys.stderr)
            continue

        filename = next_case_filename(case["id"])
        path = os.path.join(DATASET_DIR, filename)
        if args.dry_run:
            print(f"  Would write {filename} (finding_id={record['finding_id']})")
        else:
            with open(path, "w") as f:
                json.dump(case, f, indent=2)
            print(f"  Wrote {filename} (finding_id={record['finding_id']})")
        folded += 1

    print(f"\nDone: {folded} folded, {failed} skipped due to errors.")


if __name__ == "__main__":
    main()
