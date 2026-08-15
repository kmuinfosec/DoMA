"""Variation quality — surface-level (lexical) metrics only.

Computed per (original document, variant document) pair, at document level. These measure how
much the surface form changed; they make no claim about meaning, and no model is involved.

Direction:
    ngram_overlap(1/2/3), bleu, jaccard_tokens : lower = more heavily rewritten
    norm_edit_distance                         : higher = more heavily rewritten

Tokenization is deliberately simple and deterministic (lowercase ``[a-z0-9]+``), so the numbers
are reproducible without any external resource.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd
from rapidfuzz.distance import Levenshtein

# Every surface metric reported, in table order.
LEXICAL_METRIC_NAMES = [
    "ngram_overlap_1", "ngram_overlap_2", "ngram_overlap_3",
    "bleu", "norm_edit_distance", "jaccard_tokens",
]


# Lowercase alphanumeric tokens.
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


# n-gram tuples of a token sequence (empty when the sequence is shorter than n).
def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


# Fraction of the variant's distinct n-grams that also occur in the original (precision).
# Lower means more rewriting.
def ngram_overlap(ref: str, hyp: str, n: int = 1) -> float:
    ref_ng = set(_ngrams(_tokenize(ref), n))
    hyp_ng = set(_ngrams(_tokenize(hyp), n))
    if not hyp_ng:
        return 0.0
    return len(ref_ng & hyp_ng) / len(hyp_ng)


# Jaccard similarity of the two token sets. Lower means more rewriting.
def jaccard_tokens(ref: str, hyp: str) -> float:
    ref_set, hyp_set = set(_tokenize(ref)), set(_tokenize(hyp))
    union = ref_set | hyp_set
    if not union:
        return 0.0
    return len(ref_set & hyp_set) / len(union)


# Token-level edit distance normalized by the longer sequence. Higher means more rewriting.
# rapidfuzz (C accelerated); a pure-python pass over full papers is too slow.
def norm_edit_distance(ref: str, hyp: str) -> float:
    r, h = _tokenize(ref), _tokenize(hyp)
    if not r and not h:
        return 0.0
    return Levenshtein.distance(r, h) / max(len(r), len(h))


# BLEU (Papineni et al., 2002) of the variant against the original: clipped n-gram precisions for
# n = 1..4, uniform weights, geometric mean, brevity penalty. Only orders with no match at all are
# smoothed (NLTK smoothing method 1: epsilon in the numerator), so a single missing order does not
# collapse the score to zero while the matched orders keep their exact precision.
# Lower means more rewriting.
def bleu(ref: str, hyp: str, max_n: int = 4, epsilon: float = 0.1) -> float:
    ref_toks, hyp_toks = _tokenize(ref), _tokenize(hyp)
    c, r = len(hyp_toks), len(ref_toks)
    if c == 0 or r == 0:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        hyp_ng = Counter(_ngrams(hyp_toks, n))
        total = sum(hyp_ng.values())
        if not total:                        # variant shorter than n: BLEU-4 is undefined
            return 0.0
        ref_ng = Counter(_ngrams(ref_toks, n))
        overlap = sum(min(count, ref_ng.get(gram, 0)) for gram, count in hyp_ng.items())
        precisions.append((overlap if overlap else epsilon) / total)
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    brevity = 1.0 if c > r else math.exp(1 - r / c)
    return brevity * geo_mean


# Every surface-level metric for one (original, variant) pair.
def lexical_metrics(ref: str, hyp: str) -> dict[str, float]:
    return {
        "ngram_overlap_1": ngram_overlap(ref, hyp, 1),
        "ngram_overlap_2": ngram_overlap(ref, hyp, 2),
        "ngram_overlap_3": ngram_overlap(ref, hyp, 3),
        "bleu": bleu(ref, hyp),
        "norm_edit_distance": norm_edit_distance(ref, hyp),
        "jaccard_tokens": jaccard_tokens(ref, hyp),
    }


# Pair every document of a variant set with its original via source_doc_id and score them.
# One row per variant document.
def pairwise_lexical_df(original_df: pd.DataFrame, variant_df: pd.DataFrame) -> pd.DataFrame:
    ref_text = dict(zip(original_df["doc_id"], original_df["text"]))
    rows = []
    for row in variant_df.itertuples(index=False):
        ref = ref_text.get(row.source_doc_id)
        if ref is None:                      # variant whose original is absent
            continue
        rows.append({
            "doc_id": row.doc_id,
            "source_doc_id": row.source_doc_id,
            "family_id": row.family_id,
            "variant_type": row.variant_type,
            "variant_level": row.variant_level,
            **lexical_metrics(ref, row.text),
        })
    return pd.DataFrame(rows)


# Mean of every surface metric per (variant_type, variant_level) - the variation quality table.
def summarize_lexical_df(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in LEXICAL_METRIC_NAMES if c in pairwise_df.columns]
    return (pairwise_df
            .groupby(["variant_type", "variant_level"], dropna=False)[metric_cols]
            .mean()
            .join(pairwise_df.groupby(["variant_type", "variant_level"], dropna=False)
                  .size().rename("n_docs"))
            .reset_index())
