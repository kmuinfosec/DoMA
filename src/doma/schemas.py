"""Canonical schema of the document table (docs_df).

Every prepared parquet under ``data/prepared`` follows ``DOC_COLUMNS``.
"""
from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

# One document per row. Every prepared parquet carries exactly this column order.
DOC_COLUMNS: list[str] = [
    "doc_id",         # unique document id
    "dataset",        # dataset name (krapivin / par3 / casimir)
    "family_id",      # groups every document derived from one source text (unit of the 50/50 split)
    "text",           # document body
    "version_index",  # order within a family (CASIMIR revisions 1..n; PAR3 machine=0, humans 1..n; else empty)
    "variant_type",   # original | revision | dipper | human_translation | ...
    "variant_level",  # variation strength (e.g. DIPPER lex60_order60)
    "variant_seed",   # seed used to generate the variant
    "source_doc_id",  # doc_id this document was derived from (pairs a variant with its original)
    "num_chars",      # character count of text (derived)
    "num_words",      # word count of text (derived)
    "meta_json",      # dataset specific extras, JSON string
]


# Raise if any required column is missing from df.
def _ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


# Coerce a meta value into a JSON string (None -> '{}', dict -> serialized).
def _to_meta_json(value: object) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# Normalize an arbitrary docs_df into the DOC_COLUMNS schema.
# - raises when doc_id/dataset/family_id/text are missing, or when doc_id is duplicated
# - fills optional columns with defaults and derives num_chars / num_words
def normalize_docs_df(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(df, ["doc_id", "dataset", "family_id", "text"], "docs_df")

    out = df.copy()

    optional_defaults: dict[str, object] = {
        "version_index": pd.NA,
        "variant_type": "original",
        "variant_level": pd.NA,
        "variant_seed": pd.NA,
        "source_doc_id": pd.NA,
        "meta_json": "{}",
    }
    for col, default_value in optional_defaults.items():
        if col not in out.columns:
            out[col] = default_value

    out["text"] = out["text"].fillna("").astype(str)
    out["num_chars"] = out["text"].map(len)
    out["num_words"] = out["text"].map(lambda x: len(x.split()))
    out["meta_json"] = out["meta_json"].map(_to_meta_json)

    dup_mask = out["doc_id"].duplicated()
    if dup_mask.any():
        dupes = out.loc[dup_mask, "doc_id"].tolist()[:10]
        raise ValueError(f"docs_df contains duplicate doc_id: {dupes}")

    for col in DOC_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out[DOC_COLUMNS].reset_index(drop=True)
