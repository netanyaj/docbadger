"""
Precision/recall/F1 for the Verifier's stale/not-stale binary
classification -- Architecture Section 19 point 2: "Metrics tracked per
prompt version: precision, recall, F1 on staleness detection."

Computed from run_eval.py's per-case result dicts, which carry both the
Verifier's real prediction (verifier_stale) and the case's gold label
(verifier_expected_stale, added to the result dict alongside it). A pure
function, deliberately separate from run_eval.py's own I/O/API-calling
code, so it's directly unit-testable against synthetic result lists
without touching a real LLM.
"""


def compute_verifier_metrics(results: list) -> dict:
    """Only counts cases with BOTH a real prediction (verifier_stale is not
    None -- excludes cases where the LLM call itself failed) AND a gold
    label (verifier_expected_stale present) -- same "don't score what
    wasn't actually exercised" discipline as run_eval.py's own match-rate
    scoring and confidence_calibration_check.py's gold_is_accurate().
    """
    tp = fp = fn = tn = 0
    for r in results:
        predicted = r.get("verifier_stale")
        expected = r.get("verifier_expected_stale")
        if predicted is None or expected is None:
            continue
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

    counted = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "counted": counted,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
