import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from eval_metrics import compute_verifier_metrics


def _r(predicted, expected):
    return {"verifier_stale": predicted, "verifier_expected_stale": expected}


def test_perfect_classifier_has_precision_recall_f1_of_one():
    results = [_r(True, True), _r(True, True), _r(False, False)]
    m = compute_verifier_metrics(results)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["counted"] == 3


def test_false_positive_lowers_precision_not_recall():
    results = [_r(True, True), _r(True, False)]  # 1 TP, 1 FP
    m = compute_verifier_metrics(results)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 1.0


def test_false_negative_lowers_recall_not_precision():
    results = [_r(True, True), _r(False, True)]  # 1 TP, 1 FN
    m = compute_verifier_metrics(results)
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5


def test_cases_with_no_prediction_are_excluded_not_scored_as_wrong():
    # verifier_stale=None means the LLM call itself failed -- must not be
    # silently counted as a wrong prediction (would unfairly punish
    # precision/recall for an infra failure, not a model failure).
    results = [_r(True, True), _r(None, True), _r(None, False)]
    m = compute_verifier_metrics(results)
    assert m["counted"] == 1
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_cases_with_no_gold_label_are_excluded():
    results = [_r(True, True), _r(True, None)]
    m = compute_verifier_metrics(results)
    assert m["counted"] == 1


def test_empty_results_produce_none_metrics_not_a_crash():
    m = compute_verifier_metrics([])
    assert m["counted"] == 0
    assert m["precision"] is None
    assert m["recall"] is None
    assert m["f1"] is None


def test_no_positive_predictions_at_all_gives_none_precision_not_zero_division():
    results = [_r(False, False), _r(False, True)]  # 1 TN, 1 FN, zero TP+FP
    m = compute_verifier_metrics(results)
    assert m["precision"] is None
    assert m["recall"] == 0.0
