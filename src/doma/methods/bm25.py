"""BM25 baseline (bm25s) — sparse lexical retrieval over full documents.

The confidential reference documents form the BM25 corpus; each query document is scored
against it and attributed to the family with the highest score.

``confidence`` is the raw BM25 score of that best family, evaluated by a threshold sweep.
Indexing counts as registration (``build_sec``); tokenizing and scoring the query is inference.
"""
from __future__ import annotations

from time import perf_counter

import pandas as pd

from doma.matches import MATCH_COLUMNS


# Score every query document against the confidential BM25 index and keep the best family per query.
def bm25_matches(reference_docs_df, query_docs_df, k1=1.5, b=0.75, top_k=200):
    import bm25s

    ref_doc = reference_docs_df["doc_id"].to_numpy()
    ref_family = reference_docs_df["family_id"].to_numpy()
    corpus_texts = [str(t or "").strip() for t in reference_docs_df["text"].fillna("")]
    query_texts = [str(t or "").strip() for t in query_docs_df["text"].fillna("")]
    if not corpus_texts:
        return pd.DataFrame(columns=MATCH_COLUMNS), {"inference_total_sec": 0.0, "build_sec": 0.0}

    # ---- index the confidential corpus (= registration) ----
    build_started = perf_counter()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(corpus_tokens, show_progress=False)
    build_sec = perf_counter() - build_started

    # ---- tokenize and score the queries (= inference) ----
    inference_started = perf_counter()
    query_tokens = bm25s.tokenize(query_texts, stopwords="en", show_progress=False)
    k = min(len(corpus_texts), top_k)
    results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
    inference_sec = perf_counter() - inference_started

    # ---- keep the highest-scoring family per query ----
    query_ids = query_docs_df["doc_id"].to_numpy()
    query_families = query_docs_df["family_id"].to_numpy()
    rows = []
    for qi in range(len(query_docs_df)):
        best_score, best_doc, best_family = 0.0, None, None
        for idx, score in zip(results[qi], scores[qi]):
            score = float(score)
            if score > best_score:
                best_score, best_doc, best_family = score, ref_doc[idx], ref_family[idx]
        rows.append({
            "query_doc_id": query_ids[qi], "query_family_id": query_families[qi],
            "pred_doc_id": best_doc, "pred_family_id": best_family,
            "confidence": best_score,   # raw BM25 score, scored by threshold sweep
        })
    timing = {"inference_total_sec": inference_sec, "build_sec": build_sec}
    return pd.DataFrame(rows, columns=MATCH_COLUMNS), timing
