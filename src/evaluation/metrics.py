"""Evaluation metrics for financial statement verification.

Computes detection metrics (accuracy, precision, recall, F1, FPR),
localization accuracy, per-category and per-magnitude breakdowns,
and the detection threshold m_50.

All public functions accept lists of predictions and ground truths
so they can be used independently of the evaluation harness.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Each prediction dict must have at least:
#   "has_error": bool
#   "error_location": str | None
# Each ground-truth dict must have at least:
#   "has_error": bool
#   "error_location": str | None
#   "error_category": str | None      (e.g. "AE", "CL", "YOY", "MR")
#   "error_magnitude_pct": float | None
Prediction = Dict[str, Any]
GroundTruth = Dict[str, Any]


# ---------------------------------------------------------------------------
# Core binary detection metrics
# ---------------------------------------------------------------------------

def _binary_counts(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> Tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) for error detection.

    A *positive* means the instance contains an error.
    """
    tp = fp = tn = fn = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_err = bool(pred.get("has_error"))
        gt_err = bool(gt.get("has_error"))
        if gt_err and pred_err:
            tp += 1
        elif not gt_err and pred_err:
            fp += 1
        elif not gt_err and not pred_err:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def accuracy(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> float:
    """Fraction of instances where detection decision is correct."""
    tp, fp, tn, fn = _binary_counts(predictions, ground_truths)
    total = tp + fp + tn + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> float:
    """Precision: TP / (TP + FP)."""
    tp, fp, _tn, _fn = _binary_counts(predictions, ground_truths)
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> float:
    """Recall (sensitivity): TP / (TP + FN)."""
    tp, _fp, _tn, fn = _binary_counts(predictions, ground_truths)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def f1_score(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(predictions, ground_truths)
    r = recall(predictions, ground_truths)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def false_positive_rate(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> float:
    """False positive rate: FP / (FP + TN).

    Computed only over *clean* (no-error) instances.
    """
    _tp, fp, tn, _fn = _binary_counts(predictions, ground_truths)
    denom = fp + tn
    return fp / denom if denom > 0 else 0.0


def detection_metrics(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> Dict[str, float]:
    """Compute all core detection metrics in one pass.

    Returns a dict with keys: accuracy, precision, recall, f1, fpr.
    """
    tp, fp, tn, fn = _binary_counts(predictions, ground_truths)
    total = tp + fp + tn + fn

    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Localization accuracy
# ---------------------------------------------------------------------------

def _location_matches(pred_loc: Optional[str], gt_loc: Optional[str]) -> bool:
    """Check whether the predicted location matches the ground truth.

    A match requires that the predicted dot-path ends with the same
    terminal field name, allowing for prefix flexibility (e.g.
    ``"income_statement.revenue"`` matches ``"income_statement.revenue"``
    and a prediction of ``"revenue"`` also matches).
    """
    if pred_loc is None or gt_loc is None:
        return False
    pred_parts = pred_loc.strip().lower().replace(" ", "_").split(".")
    gt_parts = gt_loc.strip().lower().replace(" ", "_").split(".")
    # Exact match.
    if pred_parts == gt_parts:
        return True
    # Terminal field match (e.g. "revenue" vs "income_statement.revenue").
    if pred_parts[-1] == gt_parts[-1]:
        return True
    return False


def localization_accuracy(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> float:
    """Fraction of *correctly detected* errors where location is also correct.

    Only considers true-positive instances (model said error AND there is one).
    """
    correct = 0
    total_tp = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_err = bool(pred.get("has_error"))
        gt_err = bool(gt.get("has_error"))
        if gt_err and pred_err:
            total_tp += 1
            # Handle multi-error ground truths.
            gt_locations: List[Optional[str]] = []
            if "errors" in gt:
                gt_locations = [e.get("error_location") for e in gt["errors"]]
            else:
                gt_locations = [gt.get("error_location")]
            pred_loc = pred.get("error_location")
            if any(_location_matches(pred_loc, gl) for gl in gt_locations):
                correct += 1

    return correct / total_tp if total_tp > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-category metrics
# ---------------------------------------------------------------------------

def _get_category(gt: GroundTruth) -> Optional[str]:
    """Extract the top-level error category from a ground truth dict."""
    # Single-error instances store error_category directly.
    cat = gt.get("error_category")
    if cat:
        return cat
    # Multi-error: return the category of the first error.
    errors = gt.get("errors", [])
    if errors:
        return errors[0].get("error_category")
    return None


def per_category_metrics(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> Dict[str, Dict[str, float]]:
    """Compute detection metrics broken down by error category (AE, CL, YoY, MR).

    Returns a dict keyed by category string, each containing the standard
    metrics dict from ``detection_metrics``.  Clean (no-error) instances are
    excluded from per-category results.
    """
    buckets: Dict[str, Tuple[List[Prediction], List[GroundTruth]]] = defaultdict(
        lambda: ([], [])
    )

    for pred, gt in zip(predictions, ground_truths):
        if not gt.get("has_error"):
            continue
        cat = _get_category(gt)
        if cat is None:
            continue
        preds_list, gts_list = buckets[cat]
        preds_list.append(pred)
        gts_list.append(gt)

    results: Dict[str, Dict[str, float]] = {}
    for cat in sorted(buckets):
        preds_list, gts_list = buckets[cat]
        results[cat] = detection_metrics(preds_list, gts_list)
        # Also add detection rate (same as recall, but named explicitly).
        results[cat]["detection_rate"] = results[cat]["recall"]
    return results


def per_category_detection_rates(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> Dict[str, float]:
    """Return detection rate (recall) per error category.

    Convenience function that returns just the detection rates.
    """
    cat_metrics = per_category_metrics(predictions, ground_truths)
    return {cat: m["detection_rate"] for cat, m in cat_metrics.items()}


# ---------------------------------------------------------------------------
# Per-magnitude metrics
# ---------------------------------------------------------------------------

_MAGNITUDE_BINS = [
    ("<1%", 0.0, 1.0),
    ("1-5%", 1.0, 5.0),
    ("5-10%", 5.0, 10.0),
    ("10-20%", 10.0, 20.0),
    (">20%", 20.0, float("inf")),
]


def _magnitude_bin(pct: float) -> str:
    """Map an absolute magnitude percentage to a bin label."""
    abs_pct = abs(pct)
    for label, lo, hi in _MAGNITUDE_BINS:
        if lo <= abs_pct < hi:
            return label
    return ">20%"


def per_magnitude_metrics(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> Dict[str, Dict[str, float]]:
    """Compute detection metrics broken down by error magnitude bin.

    Only error instances are included.  Returns a dict keyed by magnitude
    bin label (e.g. ``"<1%"``, ``"1-5%"``).
    """
    buckets: Dict[str, Tuple[List[Prediction], List[GroundTruth]]] = defaultdict(
        lambda: ([], [])
    )

    for pred, gt in zip(predictions, ground_truths):
        if not gt.get("has_error"):
            continue
        mag = gt.get("error_magnitude_pct")
        if mag is None:
            # Multi-error: use the max magnitude.
            errors = gt.get("errors", [])
            mags = [abs(e.get("error_magnitude_pct", 0) or 0) for e in errors]
            mag = max(mags) if mags else 0.0
        bin_label = _magnitude_bin(mag)
        preds_list, gts_list = buckets[bin_label]
        preds_list.append(pred)
        gts_list.append(gt)

    results: Dict[str, Dict[str, float]] = {}
    for label, _lo, _hi in _MAGNITUDE_BINS:
        if label in buckets:
            preds_list, gts_list = buckets[label]
            results[label] = detection_metrics(preds_list, gts_list)
            results[label]["detection_rate"] = results[label]["recall"]
    return results


def per_magnitude_detection_rates(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> Dict[str, float]:
    """Return detection rate per magnitude bin (convenience wrapper)."""
    mag_metrics = per_magnitude_metrics(predictions, ground_truths)
    return {label: m["detection_rate"] for label, m in mag_metrics.items()}


# ---------------------------------------------------------------------------
# Detection threshold m_50
# ---------------------------------------------------------------------------

def detection_threshold_m50(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
    magnitude_points: Optional[Sequence[float]] = None,
) -> Optional[float]:
    """Estimate the error magnitude at which detection first exceeds 50%.

    Groups error instances by their exact magnitude percentage, computes the
    detection rate at each magnitude, and returns the smallest magnitude
    where detection rate >= 0.5.

    If *magnitude_points* is given, only those magnitude values are
    considered; otherwise all unique magnitudes in the data are used.

    Returns ``None`` if detection never reaches 50%.
    """
    # Collect (magnitude, detected?) pairs for error instances.
    mag_detected: Dict[float, List[bool]] = defaultdict(list)
    for pred, gt in zip(predictions, ground_truths):
        if not gt.get("has_error"):
            continue
        mag = gt.get("error_magnitude_pct")
        if mag is None:
            errors = gt.get("errors", [])
            mags = [abs(e.get("error_magnitude_pct", 0) or 0) for e in errors]
            mag = max(mags) if mags else 0.0
        abs_mag = abs(mag)
        detected = bool(pred.get("has_error"))
        mag_detected[abs_mag].append(detected)

    if magnitude_points is not None:
        candidates = sorted(set(magnitude_points))
    else:
        candidates = sorted(mag_detected.keys())

    for m in candidates:
        detections = mag_detected.get(m)
        if detections is None:
            continue
        rate = sum(detections) / len(detections)
        if rate >= 0.5:
            return m

    return None


# ---------------------------------------------------------------------------
# Convenience: full report
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: Sequence[Prediction],
    ground_truths: Sequence[GroundTruth],
) -> Dict[str, Any]:
    """Compute every metric and return as a single nested dict.

    Top-level keys:
      - ``overall``      : core detection metrics
      - ``localization``  : localization accuracy
      - ``by_category``   : per-category metrics
      - ``by_magnitude``  : per-magnitude metrics
      - ``m50``           : detection threshold
    """
    return {
        "overall": detection_metrics(predictions, ground_truths),
        "localization": round(localization_accuracy(predictions, ground_truths), 4),
        "by_category": per_category_metrics(predictions, ground_truths),
        "by_magnitude": per_magnitude_metrics(predictions, ground_truths),
        "m50": detection_threshold_m50(predictions, ground_truths),
    }
