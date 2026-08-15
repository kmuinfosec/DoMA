"""Fuzzy hash baseline (ssdeep via ppdeep) — context-triggered piecewise hashing of full documents.

Each document is reduced to a fuzzy hash; every query hash is compared against every confidential
hash (0-100) and attributed to the family with the highest similarity. There is no embedding and
no vector index, so the comparison is an O(N_ref x N_query) scan.

``confidence`` is that best similarity rescaled to 0-1, evaluated by a threshold sweep.
Hashing the references counts as registration; hashing and comparing the queries is inference.
"""
from __future__ import annotations

from time import perf_counter

import pandas as pd
from tqdm.auto import tqdm

from doma.matches import MATCH_COLUMNS


# Hash a list of texts; empty texts map to None.
def _hash_texts(texts):
    import ppdeep

    return [ppdeep.hash(t) if t else None for t in texts]


# Compare every query hash against the confidential hashes and keep the best family per query.
def ssdeep_matches(reference_docs_df, query_docs_df):
    import ppdeep

    ref_doc = reference_docs_df["doc_id"].to_numpy()
    ref_family = reference_docs_df["family_id"].to_numpy()
    ref_texts = [str(t or "").strip() for t in reference_docs_df["text"].fillna("")]

    # ---- hash the confidential documents (= registration) ----
    build_started = perf_counter()
    ref_hashes = _hash_texts(ref_texts)
    ref_entries = [(h, ref_doc[i], ref_family[i]) for i, h in enumerate(ref_hashes) if h]
    build_sec = perf_counter() - build_started

    # ---- hash and compare the queries (= inference) ----
    query_ids = query_docs_df["doc_id"].to_numpy()
    query_families = query_docs_df["family_id"].to_numpy()
    query_texts = [str(t or "").strip() for t in query_docs_df["text"].fillna("")]
    inference_started = perf_counter()
    query_hashes = _hash_texts(query_texts)
    rows = []
    for qi in tqdm(range(len(query_docs_df)), desc="ssdeep compare", leave=False):
        query_hash = query_hashes[qi]
        best_score, best_doc, best_family = 0, None, None
        if query_hash:
            for ref_hash, doc_id, family_id in ref_entries:
                score = ppdeep.compare(query_hash, ref_hash)
                if score > best_score:
                    best_score, best_doc, best_family = score, doc_id, family_id
        rows.append({
            "query_doc_id": query_ids[qi], "query_family_id": query_families[qi],
            "pred_doc_id": best_doc, "pred_family_id": best_family,
            "confidence": float(best_score) / 100.0,   # ppdeep reports 0-100
        })
    inference_sec = perf_counter() - inference_started
    timing = {"inference_total_sec": inference_sec, "build_sec": build_sec}
    return pd.DataFrame(rows, columns=MATCH_COLUMNS), timing
