"""
Derives confidence-rubric inputs (change_type, link_source) for an isolated
(old_code, new_code, doc_section) triple, by reusing the SAME rules the
real pipeline applies -- never a re-guessed approximation. Built for the
eval harness's confidence-tier calibration check (Architecture Section 19
point 4), which needs the real rubric's tier assignment per eval case but
has no real repo/index to run the full indexing pipeline against.

This limitation is not new or invented for this fix -- several eval case
files (014, 015, 020, 021) already carry notes_for_labeler entries flagging
that run_eval.py's stage-level harness never exercises indexer.py's real
linking or confidence_rubric.py's real tier scoring, and that confirming
those end-to-end requires a real repo/PR run (Milestone 7). This module
narrows that gap on the link_source side specifically: instead of leaving
it as an untested unknown, it derives what the real heuristic_linker WOULD
find, by applying its exact matching rule (doc_parser._extract_mentions +
qualified/leaf name comparison) to the case's own text. It does not run
embedding_linker at all (that requires a real embedding model call), so a
"no heuristic match" result is reported as link_source="embedding" as an
assumption -- true for every case in this dataset, since each one is
constructed specifically to test a scenario where the pipeline is
expected to find and process the change -- not as a confirmed embedding
hit. This assumption is disclosed everywhere this module's output is used,
never presented as a validated finding.

blast_radius is NOT derived here -- it depends on the full doc set a code
change is linked against, which one isolated eval case can't supply. It's
a hand-authored label on each case file instead (labels.blast_radius).
"""

from doc_parser import _extract_mentions
from diff_analyzer import _first_def


def infer_link_source(function_name: str, doc_text: str) -> str:
    """function_name: the modified function/method's own qualified name,
    e.g. "GZipMiddleware.__init__" or "send_email" (top-level functions
    have no class prefix, so their qualified name equals their leaf name).

    Mirrors heuristic_linker.build_heuristic_links_with_source's exact/leaf
    logic exactly, applied to a single (function, doc) pair instead of a
    whole-repo index:
      - 'exact' if the doc mentions the full qualified name verbatim, in a
        backtick-wrapped inline code span (doc_parser.MENTION_PATTERN only
        scans those -- NOT plain prose text and NOT fenced ``` code blocks,
        which are invisible to mention extraction entirely, however
        obviously the name appears there to a human reader).
      - 'leaf' if only the bare name (the part after the last '.') is
        mentioned, not the full qualified form -- e.g. a doc mentioning
        `close()` links to any chunk whose own name ends in ".close" (or is
        exactly "close"), not specifically to one class's method.
      - 'embedding' if neither is mentioned at all -- the real heuristic
        stage would find nothing for this chunk, so it would only ever be
        found (if at all) by the embedding-similarity fallback.
    """
    mentions = _extract_mentions(doc_text)
    if function_name in mentions:
        return "exact"
    leaf = function_name.rsplit(".", 1)[-1]
    if any(m.rsplit(".", 1)[-1] == leaf for m in mentions):
        return "leaf"
    return "embedding"


def derive_confidence_inputs(old_code: str, new_code: str, doc_text: str) -> dict:
    """Convenience wrapper combining change_type (diff_analyzer.
    classify_change_type) and link_source (infer_link_source above) for
    one eval case. Does not include blast_radius -- see module docstring.
    """
    from diff_analyzer import classify_change_type

    function_name, _ = _first_def(new_code)
    return {
        "change_type": classify_change_type(old_code, new_code),
        "link_source": infer_link_source(function_name, doc_text),
    }
