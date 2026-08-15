"""Vector search — top-k nearest confidential documents for each query document.

One (query document, reference document) pair per row; ``matches.best_match_by_document``
then reduces this to one row per query.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Long-form search result schema (kept identical even when the result is empty).
RETRIEVAL_COLUMNS = [
    "query_doc_id", "query_family_id",
    "ref_doc_id", "ref_family_id",
    "score", "rank",
]


# Search the index with query document vectors and return the (query, reference) pairs long-form.
def search_index(
    index_artifact,
    query_doc_ids,
    query_family_ids,
    query_embeddings: np.ndarray,
    top_k: int = 1,
) -> pd.DataFrame:
    if len(query_doc_ids) != len(query_embeddings):
        raise ValueError("query_doc_ids and query_embeddings must have the same length")
    if len(query_doc_ids) == 0:
        return pd.DataFrame(columns=RETRIEVAL_COLUMNS)

    scores, indices = index_artifact.search(query_embeddings.astype(np.float32), top_k)
    ref = index_artifact.meta_df.reset_index(drop=True)
    ref_doc_ids = ref["doc_id"].to_numpy()
    ref_family_ids = ref["family_id"].to_numpy()

    rows: list[dict] = []
    for q_idx, (q_doc_id, q_family_id) in enumerate(zip(query_doc_ids, query_family_ids)):
        for rank, (idx, score) in enumerate(zip(indices[q_idx], scores[q_idx]), start=1):
            if idx < 0:   # faiss returns -1 when fewer than top_k neighbours exist
                continue
            rows.append({
                "query_doc_id": q_doc_id,
                "query_family_id": q_family_id,
                "ref_doc_id": ref_doc_ids[idx],
                "ref_family_id": ref_family_ids[idx],
                "score": float(score),
                "rank": rank,
            })
    return pd.DataFrame(rows, columns=RETRIEVAL_COLUMNS)
