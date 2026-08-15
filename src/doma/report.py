"""Result scan and summary tables — reads every ``runs/**/metrics.json`` into one frame.

``metrics.json`` is the single source of truth; no summary file is written, the tables are
always rebuilt by scanning. Layout: ``artifacts/runs/<run_ident>/<run_tag>/metrics.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Row order of the result table: (label, method, run_tag substring or None for the whole method).
PAPER_ROWS: list[tuple[str, str, str | None]] = [
    ("DoMA (Qwen3 0.6B)", "doma", "qwen3"),
    ("DoMA (Granite 97M)", "doma", "granite"),
    ("DoMA (BGE 109M)", "doma", "bge"),
    ("BM25", "bm25", None),
    ("Fuzzy hash", "ssdeep", None),
]

# Column order and display labels of the result table.
PAPER_COLUMNS: list[tuple[str, str]] = [
    ("krapivin", "Krapivin paraphrase"),
    ("par3", "PAR3 translation"),
    ("casimir_raw", "CASIMIR revision"),
    ("casimir_dedup", "CASIMIR revision (dedup)"),
]

# The two metrics shown in the result table.
PAPER_METRICS = {
    "best_f1": "Best-F1 under an oracle threshold",
    "attribution": "Source-attribution accuracy",
}


# Registration (offline) cost = encoding/hashing the references + building the index.
# Per the latency rule this is excluded from inference and reported separately.
def _registration_sec(t: dict):
    build = t.get("build_sec")
    if build is None:
        return None
    return float(build)


# Scan every metrics.json under artifacts/runs into one flat DataFrame.
def load_all_metrics(artifacts_dir: str | Path = "artifacts") -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(Path(artifacts_dir, "runs").glob("*/*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        detection = m.get("detection", {})
        best = m.get("detection_best", {})
        attribution = m.get("attribution", {})
        separability = m.get("separability", {})
        timing = m.get("timing_sec", {})
        counts = m.get("counts", {})
        rows.append({
            "method": m.get("method"),
            "dataset": m.get("dataset"),
            "variant_sets": "|".join(m.get("variant_sets") or []),
            "inclorig": m.get("include_original_as_positive"),
            "run_ident": path.parent.parent.name,
            "run_tag": path.parent.name,
            "best_f1": best.get("best_f1"),
            "best_threshold": best.get("best_threshold"),
            "best_precision": best.get("best_precision"),
            "best_recall": best.get("best_recall"),
            # source-attribution accuracy: correct family over *all* positive queries
            "attribution": attribution.get("family_acc_on_all_positive"),
            "attribution_on_detected": attribution.get("family_acc_on_detected_positive"),
            "f1": detection.get("f1"),
            "roc_auc": separability.get("roc_auc"),
            "pr_auc": separability.get("pr_auc"),
            "inference_sec": timing.get("inference_total_sec"),
            "registration_sec": _registration_sec(timing),
            "n_ref_docs": counts.get("n_ref_docs"),
            "n_query_docs": counts.get("n_query_docs"),
        })
    return pd.DataFrame(rows)


# Keep only the runs that match the experimental axes for one dataset.
# A cell of the table is one specific experiment, not "whatever ran on this dataset": the variant
# combination and include_original_as_positive are part of its identity. Without this filter a run
# on a different axis would silently compete for the same cell.
def select_axis(df: pd.DataFrame, dataset: str, variant_sets, include_original_as_positive: bool):
    from doma.pipeline.config import DATASET_DEFAULTS

    expected = tuple(variant_sets if variant_sets is not None
                     else DATASET_DEFAULTS[dataset]["variant_sets"])
    return df[(df["dataset"] == dataset)
              & (df["variant_sets"] == "|".join(expected))
              & (df["inclorig"] == include_original_as_positive)]


# Build one result table: rows = methods (model variants separated by run_tag), columns = datasets.
# A cell that was never run stays NaN; a dataset with no run at all is dropped as a column, since
# an all-NaN column carries no information about the experiment.
#
# Every cell must resolve to exactly one run. If several distinct runs still match - two split
# seeds, or the same model at two window lengths, both matching the row's tag substring - that is
# an ambiguity in the artifacts, not something to resolve by taking the best number, so it raises.
def paper_table(df: pd.DataFrame, value: str = "best_f1",
                rows_spec: list[tuple[str, str, str | None]] = PAPER_ROWS,
                columns_spec: list[tuple[str, str]] = PAPER_COLUMNS,
                include_original_as_positive: bool = False,
                variant_sets: dict[str, tuple[str, ...]] | None = None) -> pd.DataFrame:
    variant_sets = variant_sets or {}
    present = set(df["dataset"].dropna().unique())
    columns = [(key, label) for key, label in columns_spec if key in present]
    out = pd.DataFrame(index=[label for label, _, _ in rows_spec],
                       columns=[label for _, label in columns], dtype=float)
    for row_label, method, tag_substring in rows_spec:
        sub = df[df["method"] == method]
        if tag_substring:
            sub = sub[sub["run_tag"].str.contains(tag_substring, case=False, regex=False, na=False)]
        for key, column_label in columns:
            cell = select_axis(sub, key, variant_sets.get(key), include_original_as_positive)
            cell = cell[cell[value].notna()]
            if cell.empty:
                continue
            runs = cell[["run_ident", "run_tag"]].drop_duplicates()
            if len(runs) > 1:
                listed = "\n  ".join(f"{r.run_ident}/{r.run_tag}" for r in runs.itertuples())
                raise ValueError(
                    f"cell '{row_label}' x '{column_label}' matches {len(runs)} distinct runs, "
                    f"so the reported value is ambiguous:\n  {listed}\n"
                    f"Narrow the row's run_tag substring in PAPER_ROWS, or remove the stale runs.")
            out.loc[row_label, column_label] = float(cell[value].iloc[0])
    return out


# Corpus table: documents, variants, and the share of source documents that do not fit the
# encoder's context window in one pass. That share depends on the tokenizer, so the encoder is an
# explicit argument and is returned alongside the table. The counts are read from the embedding
# cache (``n_overflow`` / ``n_docs``); a set never encoded with this model reports NA.
def corpus_table(prepared_dir: str | Path = "data/prepared",
                 artifacts_dir: str | Path = "artifacts",
                 model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 dtype: str = "bfloat16") -> tuple[pd.DataFrame, dict]:
    from doma.ids import model_slug
    from doma.io import concat_prepared_sets, load_prepared_set
    from doma.pipeline.config import DATASET_DEFAULTS

    cache_root = Path(artifacts_dir) / "cache" / "embeddings_doma" / model_slug(model_name)
    window_path = cache_root / "max_seq_length.json"
    window = (int(json.loads(window_path.read_text(encoding="utf-8"))["max_seq_length"])
              if window_path.exists() else None)

    rows: list[dict] = []
    for dataset, defaults in DATASET_DEFAULTS.items():
        original_set = defaults["original_set"]
        try:
            n_docs = len(load_prepared_set(prepared_dir, original_set))
            n_variants = len(concat_prepared_sets(prepared_dir, defaults["variant_sets"]))
        except FileNotFoundError:
            continue

        over = pd.NA
        if window is not None:
            info_path = cache_root / f"{original_set}__L{window}__{dtype}.meta.json"
            if info_path.exists():
                info = json.loads(info_path.read_text(encoding="utf-8"))
                cached_docs = int(info.get("n_docs", 0))
                if cached_docs:
                    over = 100.0 * int(info.get("n_overflow", 0)) / cached_docs
        rows.append({"dataset": dataset, "original_set": original_set, "docs": n_docs,
                     "variants": n_variants, "over_window_pct": over})

    return pd.DataFrame(rows), {"encoder": model_name, "window": window, "dtype": dtype}


# Both metric blocks of the result table (best_f1 and attribution), keyed by caption.
def paper_tables(artifacts_dir: str | Path = "artifacts", digits: int = 3,
                 include_original_as_positive: bool = False,
                 variant_sets: dict[str, tuple[str, ...]] | None = None) -> dict[str, pd.DataFrame]:
    df = load_all_metrics(artifacts_dir)
    return {caption: paper_table(df, value,
                                 include_original_as_positive=include_original_as_positive,
                                 variant_sets=variant_sets).round(digits)
            for value, caption in PAPER_METRICS.items()}
