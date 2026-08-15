"""Evaluation metrics — detection (P/R/F1/Acc), source attribution, and threshold-free PR/ROC-AUC.

Timing and counts are not computed here; the pipeline merges them into the metrics dict
afterwards.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from doma.matches import apply_threshold


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


# Sweep every confidence threshold and return the point with the highest F1.
# Isolates score separability from fixed-threshold choice. Needs both classes present.
def _best_f1_sweep(labels, scores) -> dict:
    if not len(labels) or not (0 < labels.sum() < len(labels)):
        return {"best_threshold": None, "best_f1": None, "best_precision": None, "best_recall": None}
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)   # last point (recall=0) has no threshold
    i = int(f1[:-1].argmax()) if len(thresholds) else 0
    return {
        "best_threshold": float(thresholds[i]),
        "best_f1": float(f1[i]),
        "best_precision": float(precision[i]),
        "best_recall": float(recall[i]),
    }


# Ground truth per query: a positive query's own family is the answer, a benign query has none.
def build_query_manifest(positive_docs_df: pd.DataFrame, benign_docs_df: pd.DataFrame) -> pd.DataFrame:
    pos = [
        {"query_doc_id": r.doc_id, "is_positive": True, "target_family_id": r.family_id}
        for r in positive_docs_df.itertuples(index=False)
    ]
    neg = [
        {"query_doc_id": r.doc_id, "is_positive": False, "target_family_id": None}
        for r in benign_docs_df.itertuples(index=False)
    ]
    return pd.DataFrame(pos + neg)


# Apply the threshold to the match table, compare against the manifest, and return
# (aggregate metrics dict, per-query merged df).
# threshold=None (default) uses the best-F1 sweep point; pass a number to inspect a fixed operating point.
def evaluate_run(
    query_manifest_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    threshold: float | None = None,
    confidence_col: str = "confidence",
) -> tuple[dict[str, Any], pd.DataFrame]:
    # Threshold-free scores/labels for separability and the sweep (a missing match scores 0).
    scored = query_manifest_df.merge(matches_df[["query_doc_id", confidence_col]],
                                     on="query_doc_id", how="left")
    label = scored["is_positive"].astype(int).to_numpy()
    score = scored[confidence_col].fillna(0.0).to_numpy()
    best = _best_f1_sweep(label, score)
    if threshold is None:
        threshold = best["best_threshold"] if best["best_threshold"] is not None else 0.5

    pred = apply_threshold(matches_df, threshold=threshold, confidence_col=confidence_col)
    merged = query_manifest_df.merge(pred, on="query_doc_id", how="left")
    merged["pred_detected"] = merged["pred_detected"].fillna(False)

    yt = merged["is_positive"].astype(int).to_numpy()
    yp = merged["pred_detected"].astype(int).to_numpy()
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    # Source attribution: was a positive query traced back to the right confidential family?
    pos_mask = merged["is_positive"] == True
    det_pos_mask = pos_mask & (merged["pred_detected"] == True)
    fam_ok = merged["pred_family_id"] == merged["target_family_id"]
    attr_all = _safe_div(int((det_pos_mask & fam_ok).sum()), int(pos_mask.sum()))
    attr_detected = _safe_div(int((det_pos_mask & fam_ok).sum()), int(det_pos_mask.sum()))

    pr_auc = roc_auc = None
    if len(label) and 0 < label.sum() < len(label):
        pr_auc = float(average_precision_score(label, score))
        roc_auc = float(roc_auc_score(label, score))

    metrics: dict[str, Any] = {
        "threshold": float(threshold),   # operating point actually used (defaults to best_threshold)
        "confidence_col": confidence_col,
        "detection": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        },
        "detection_best": best,   # oracle sweep point (identical to `detection` when threshold=None)
        "attribution": {
            # "source-attribution accuracy": denominator = every positive query
            "family_acc_on_all_positive": attr_all,
            "family_acc_on_detected_positive": attr_detected,
        },
        "separability": {
            "pr_auc": pr_auc, "roc_auc": roc_auc,
            "positive_ratio": float(label.mean()) if len(label) else 0.0,
        },
    }
    return metrics, merged


# Per-family diagnostic table: which confidential documents are detected, attributed, or attract
# false positives.
def build_errors_by_family(eval_df: pd.DataFrame) -> pd.DataFrame:
    pos = eval_df[eval_df["is_positive"] == True].copy()
    neg = eval_df[eval_df["is_positive"] == False].copy()

    pos["_detected"] = (pos["pred_detected"] == True).astype(int)
    pos["_attr_ok"] = (
        (pos["pred_detected"] == True) & (pos["pred_family_id"] == pos["target_family_id"])
    ).astype(int)
    pos_grp = pos.groupby("target_family_id", dropna=True).agg(
        n_positive_total=("query_doc_id", "size"),
        n_positive_detected=("_detected", "sum"),
        n_attribution_correct=("_attr_ok", "sum"),
    )

    # How often a benign query was wrongly pulled toward a given family (false attraction).
    neg_det = neg[neg["pred_detected"] == True]
    if len(neg_det):
        neg_grp = neg_det.groupby("pred_family_id", dropna=True).size().rename("n_false_attraction").to_frame()
    else:
        neg_grp = pd.DataFrame(columns=["n_false_attraction"])

    out = pos_grp.join(neg_grp, how="outer").fillna(0).astype(int)
    out.index.name = "family_id"
    out = out.reset_index()
    out["detection_rate"] = out.apply(lambda r: _safe_div(r["n_positive_detected"], r["n_positive_total"]), axis=1)
    out["attribution_rate"] = out.apply(lambda r: _safe_div(r["n_attribution_correct"], r["n_positive_detected"]), axis=1)
    return out.sort_values(["n_positive_total", "n_false_attraction"], ascending=[False, False]).reset_index(drop=True)


# Benign queries that were flagged = false positives, paired with what they were pulled toward.
# Sorted by confidence descending, worst offender first.
def build_false_positive_pairs(eval_df: pd.DataFrame) -> pd.DataFrame:
    fp = eval_df[(eval_df["is_positive"] == False) & (eval_df["pred_detected"] == True)].copy()
    cols = ["query_doc_id", "query_family_id", "pred_doc_id", "pred_family_id", "confidence"]
    return fp[cols].sort_values("confidence", ascending=False).reset_index(drop=True)
