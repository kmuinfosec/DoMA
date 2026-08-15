"""Match table — the common output contract shared by every method.

Each method (DoMA, BM25, fuzzy hash) turns a query document into a single row:
which confidential family it looks most like, and how strongly (``confidence``).
``evaluate_run`` scores every method from this same table, so all methods are scored by the
same code.

``confidence`` is the method's own score (cosine / BM25 score / ssdeep similarity rescaled to
0-1). It is never calibrated across methods — every number is produced by a per-method threshold sweep.
"""
from __future__ import annotations

import pandas as pd

# Match table schema (kept identical even for an empty result).
MATCH_COLUMNS = [
    "query_doc_id",     # the query document being screened
    "query_family_id",  # its true family (diagnostics only, never used for prediction)
    "pred_doc_id",      # nearest confidential document
    "pred_family_id",   # family of that document = predicted source
    "confidence",       # raw method score for that best match
]


# Reduce a long-form retrieval table (one row per query/reference pair) to one row per query
# document: the single highest-scoring reference and its score.
def best_match_by_document(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    if len(retrieval_df) == 0:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    rows: list[dict] = []
    for query_doc_id, g in retrieval_df.groupby("query_doc_id", sort=False):
        best = g.loc[g["score"].idxmax()]
        rows.append({
            "query_doc_id": query_doc_id,
            "query_family_id": g["query_family_id"].iloc[0],
            "pred_doc_id": best["ref_doc_id"],
            "pred_family_id": best["ref_family_id"],
            "confidence": float(best["score"]),
        })
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


# Mark rows below the threshold as "not detected" and blank out their prediction.
# Returns a copy with an added pred_detected column.
def apply_threshold(
    matches_df: pd.DataFrame, threshold: float, confidence_col: str = "confidence"
) -> pd.DataFrame:
    out = matches_df.copy()
    if len(out) == 0:
        out["pred_detected"] = pd.Series(dtype=bool)
        return out
    out["pred_detected"] = out[confidence_col] >= threshold
    low = ~out["pred_detected"]
    out.loc[low, "pred_doc_id"] = None
    out.loc[low, "pred_family_id"] = None
    return out
