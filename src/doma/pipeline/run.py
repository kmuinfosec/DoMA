"""run(cfg) — the single entry point for every experiment.

Flow: family split -> reference/positive/benign query sets -> method -> evaluation -> artifacts.
All three methods go through this same path, so latency, F1, attribution and the per-query
case tables are directly comparable.

Artifacts land in ``artifacts/runs/<run_ident>/<run_tag>/``:
    config.json  metrics.json  matches.parquet  query_manifest.parquet
    per_query_eval.parquet  errors_by_family.parquet  fp_pairs.parquet  splits_used.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from doma.ids import sanitize_piece
from doma.instrument import timer
from doma.io import ensure_dir, save_json, save_parquet
from doma.metrics.core import (
    build_errors_by_family,
    build_false_positive_pairs,
    build_query_manifest,
    evaluate_run,
)
from doma.pipeline.config import RunConfig
from doma.splits.core import build_run_query_sets, make_split


# Output directory of one run: runs/<run_ident>/<run_tag>/.
# run_tag is the method's own parameter leaf (e.g. "<model-slug>__L<window>").
def run_dir_for(cfg: RunConfig, method_name: str, run_tag: str = "") -> Path:
    leaf = sanitize_piece(run_tag) if run_tag else "default"
    return Path(cfg.artifacts_dir) / "runs" / cfg.run_ident(method_name) / leaf


# Evaluate a match table against the manifest and write the evaluation artifacts of the run
# (config.json and splits_used.parquet are written by run()).
# Kept free of dataset IO so it can be exercised on synthetic frames.
def save_run(run_dir, manifest_df, matches_df, timing, method_name, counts, dataset=None,
             variant_sets=None, include_original_as_positive=None) -> dict:
    run_dir = ensure_dir(run_dir)
    metrics, eval_df = evaluate_run(manifest_df, matches_df)   # threshold=None -> oracle best-F1
    metrics["method"], metrics["dataset"] = method_name, dataset
    metrics["variant_sets"] = list(variant_sets or [])
    metrics["include_original_as_positive"] = include_original_as_positive
    metrics["timing_sec"] = dict(timing)
    metrics["counts"] = dict(counts)

    save_parquet(matches_df, run_dir / "matches.parquet")
    save_parquet(manifest_df, run_dir / "query_manifest.parquet")
    save_parquet(eval_df, run_dir / "per_query_eval.parquet")
    save_parquet(build_errors_by_family(eval_df), run_dir / "errors_by_family.parquet")
    save_parquet(build_false_positive_pairs(eval_df), run_dir / "fp_pairs.parquet")
    save_json(metrics, run_dir / "metrics.json")
    return metrics


# Execute one configured run and return its metrics dict.
def run(cfg: RunConfig) -> dict:
    from doma.methods import METHOD_BUILDERS

    method_fn, run_tag = METHOD_BUILDERS[cfg.method](cfg)
    out_dir = ensure_dir(run_dir_for(cfg, cfg.method, run_tag))
    save_json(cfg.as_serializable(), out_dir / "config.json")

    with timer() as t_total:
        split = make_split(cfg.prepared_dir, cfg.splits_dir, cfg.dataset,
                           cfg.resolved_original_set, cfg.split_seed, cfg.split_ratio)
        save_parquet(split, out_dir / "splits_used.parquet")
        query_sets = build_run_query_sets(cfg.prepared_dir, split, cfg.resolved_original_set,
                                          cfg.resolved_variant_sets, cfg.include_original_as_positive,
                                          strict=True)
        manifest = build_query_manifest(query_sets.positive_df, query_sets.benign_df)
        query_df = pd.concat([query_sets.positive_df, query_sets.benign_df], ignore_index=True)

        matches, timing = method_fn(query_sets.reference_df, query_df)

    counts = {
        "n_ref_docs": int(len(query_sets.reference_df)),
        "n_query_docs": int(len(query_sets.positive_df) + len(query_sets.benign_df)),
        "n_positive": int(len(query_sets.positive_df)),
        "n_benign": int(len(query_sets.benign_df)),
    }
    timing = {**timing, "total_sec": t_total.sec}
    return save_run(out_dir, manifest, matches, timing, cfg.method, counts,
                    dataset=cfg.dataset, variant_sets=cfg.resolved_variant_sets,
                    include_original_as_positive=cfg.include_original_as_positive)
