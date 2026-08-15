"""Command line entry point.

    python -m doma.cli prepare krapivin --src data/raw/krapivin_test.json
    python -m doma.cli prepare par3     --src data/raw/par3.pkl
    python -m doma.cli prepare casimir                      # downloads from HuggingFace
    python -m doma.cli run configs/krapivin.yaml            # every run listed in the config
    python -m doma.cli table                                # rebuild the result tables
    python -m doma.cli corpus                               # corpus sizes and over-window share
    python -m doma.cli figure                               # the ROC panels
    python -m doma.cli dipper krapivin --quant none --batch 4   # regenerate the paraphrases (GPU)
    python -m doma.cli quality krapivin_original krapivin_original_dipper_lex60_order60_s42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from doma.io import load_prepared_set, load_yaml
from doma.pipeline.config import RunConfig


# Turn one config file into the list of RunConfig objects it describes.
# Everything except `runs` is shared by all of them; each `runs` entry sets method and params.
def configs_from_file(path: str | Path) -> list[RunConfig]:
    spec = dict(load_yaml(path))
    runs = spec.pop("runs", [])
    if not runs:
        raise ValueError(f"{path}: no 'runs' entries")
    if spec.get("variant_sets") is not None:
        spec["variant_sets"] = tuple(spec["variant_sets"])
    return [RunConfig(**spec, method=r["method"], method_params=dict(r.get("params") or {}))
            for r in runs]


# `run`: execute every run of a config file, printing one result line each.
def _cmd_run(args) -> None:
    from doma.pipeline.run import run

    for cfg in configs_from_file(args.config):
        label = f"{cfg.method:8s} {cfg.dataset}"
        try:
            metrics = run(cfg)
        except Exception as exc:   # keep the matrix going when one cell cannot run
            print(f"{label}  SKIPPED  {type(exc).__name__}: {exc}")
            continue
        best = metrics["detection_best"]
        attribution = metrics["attribution"]["family_acc_on_all_positive"]
        print(f"{label}  best_f1={best['best_f1']:.3f}  attribution={attribution:.3f}  "
              f"inference={metrics['timing_sec'].get('inference_total_sec', 0):.1f}s")


# `table`: rescan artifacts/runs and print (optionally save) the result tables.
def _cmd_table(args) -> None:
    from doma.report import paper_tables

    tables = paper_tables(args.artifacts, include_original_as_positive=args.inclorig)
    for caption, table in tables.items():
        print(f"\n{caption}")
        print(table.to_string())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat({c: t for c, t in tables.items()}).to_csv(out)
        print(f"\nsaved: {out}")


# `corpus`: dataset sizes and the share of documents exceeding the encoder's context window.
def _cmd_corpus(args) -> None:
    from doma.report import corpus_table

    table, encoder = corpus_table(args.prepared, args.artifacts, args.model, args.dtype)
    if table.empty:
        print("no prepared sets found under", args.prepared)
        return
    print(f"over-window share measured with {encoder['encoder']} "
          f"(window {encoder['window']}, {encoder['dtype']})\n")
    print(table.to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"\nsaved: {out}")


# `figure`: redraw the ROC panels from the per-query scores of the recorded runs.
def _cmd_figure(args) -> None:
    from doma.figures import roc_figure

    aurocs = roc_figure(args.artifacts, args.out)
    if not aurocs:
        print("no runs matched the figure panels — nothing drawn")
        return
    for (dataset, label), auroc in aurocs.items():
        print(f"{dataset:16s} {label:22s} AUROC={auroc:.3f}")
    print(f"\nsaved: {args.out}")


# `prepare`: build the prepared parquet sets of one dataset from its raw source.
def _cmd_prepare(args) -> None:
    if args.dataset == "krapivin":
        from doma.datasets.krapivin import build_and_save_krapivin_original

        build_and_save_krapivin_original(args.src, args.prepared)
        print(f"wrote krapivin_original to {args.prepared}")
    elif args.dataset == "par3":
        from doma.datasets.par3 import build_and_save_par3_prepared_sets

        build_and_save_par3_prepared_sets(args.src, args.prepared)
        print(f"wrote par3_* sets to {args.prepared}")
    elif args.dataset == "casimir":
        from doma.datasets.casimir import build_and_save_casimir_prepared_sets

        counts = build_and_save_casimir_prepared_sets(args.prepared)
        for name, n in counts.items():
            print(f"{name}: {n}")


# `dipper`: rewrite <dataset>_original with DIPPER and store the variant set (GPU).
def _cmd_dipper(args) -> None:
    from doma.variation.dipper import (
        DipperConfig,
        DipperParaphraser,
        build_and_save_dipper_variant_set,
    )

    original = load_prepared_set(args.prepared, f"{args.dataset}_original")
    if args.limit:
        original = original.head(args.limit).reset_index(drop=True)
    cfg = DipperConfig(lex_diversity=args.lex, order_diversity=args.order, seed=args.seed,
                       max_input_tokens=args.max_input, max_new_tokens=args.max_new)
    print(f"[dipper] {args.dataset}_original  docs={len(original)}  quant={args.quant}  "
          f"batch={args.batch}  lex={args.lex}  order={args.order}  seed={args.seed}")

    paraphraser = DipperParaphraser(quant=args.quant)

    def paraphrase_fn(texts: list[str]) -> list[str]:
        if args.batch > 1:
            return paraphraser.paraphrase_batch(texts, cfg)
        return [paraphraser.paraphrase(t, cfg) for t in texts]

    name = build_and_save_dipper_variant_set(
        original, args.prepared, args.dataset, paraphrase_fn,
        lex=args.lex, order=args.order, seed=args.seed, batch_size=args.batch)
    print(f"wrote {name} to {args.prepared}")


# `quality`: surface-level variation quality of a variant set against its original.
def _cmd_quality(args) -> None:
    from doma.variation.quality import pairwise_lexical_df, summarize_lexical_df

    original = load_prepared_set(args.prepared, args.original_set)
    variant = load_prepared_set(args.prepared, args.variant_set)
    pairwise = pairwise_lexical_df(original, variant)
    print(summarize_lexical_df(pairwise).to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pairwise.to_parquet(out, index=False)
        print(f"\nper-document scores saved: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doma", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="execute every run described by a config file")
    p_run.add_argument("config", help="path to a YAML config under configs/")
    p_run.set_defaults(func=_cmd_run)

    p_table = sub.add_parser("table", help="rebuild the result tables from artifacts/runs")
    p_table.add_argument("--artifacts", default="artifacts")
    p_table.add_argument("--out", default=None, help="optional CSV output path")
    p_table.add_argument("--inclorig", action="store_true",
                         help="report the runs where the confidential originals were also "
                              "screened as positives (verbatim leak) instead of the default")
    p_table.set_defaults(func=_cmd_table)

    p_corpus = sub.add_parser("corpus", help="dataset sizes and the over-window document share")
    p_corpus.add_argument("--prepared", default="data/prepared")
    p_corpus.add_argument("--artifacts", default="artifacts")
    p_corpus.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B",
                          help="encoder whose cached embedding metadata supplies the over-window share")
    p_corpus.add_argument("--dtype", default="bfloat16")
    p_corpus.add_argument("--out", default=None, help="optional CSV output path")
    p_corpus.set_defaults(func=_cmd_corpus)

    p_figure = sub.add_parser("figure", help="redraw the ROC panels from the recorded runs")
    p_figure.add_argument("--artifacts", default="artifacts")
    p_figure.add_argument("--out", default="artifacts/figures/roc.pdf")
    p_figure.set_defaults(func=_cmd_figure)

    p_prep = sub.add_parser("prepare", help="build prepared parquet sets from a raw source")
    p_prep.add_argument("dataset", choices=["krapivin", "par3", "casimir"])
    p_prep.add_argument("--src", default=None, help="raw source file (not needed for casimir)")
    p_prep.add_argument("--prepared", default="data/prepared")
    p_prep.set_defaults(func=_cmd_prepare)

    p_dipper = sub.add_parser("dipper", help="generate a DIPPER paraphrase variant set (GPU)")
    p_dipper.add_argument("dataset", help="reads <dataset>_original, writes "
                                          "<dataset>_original_dipper_lex<L>_order<O>_s<seed>")
    p_dipper.add_argument("--prepared", default="data/prepared")
    p_dipper.add_argument("--quant", default="none", choices=["none", "8bit", "4bit"],
                          help="none = fp16; 4bit/8bit trade throughput for memory")
    p_dipper.add_argument("--batch", type=int, default=1,
                          help="documents rewritten per generate call. >1 seeds sampling once per "
                               "batch, so the output depends on batch composition")
    p_dipper.add_argument("--lex", type=int, default=60, help="lexical diversity, 0-100 by 20")
    p_dipper.add_argument("--order", type=int, default=60, help="order diversity, 0-100 by 20")
    p_dipper.add_argument("--seed", type=int, default=42)
    p_dipper.add_argument("--max-input", dest="max_input", type=int, default=512)
    p_dipper.add_argument("--max-new", dest="max_new", type=int, default=512)
    p_dipper.add_argument("--limit", type=int, default=0,
                          help="rewrite only the first N documents (0 = all), for a timing probe")
    p_dipper.set_defaults(func=_cmd_dipper)

    p_quality = sub.add_parser("quality", help="surface-level variation quality of a variant set")
    p_quality.add_argument("original_set")
    p_quality.add_argument("variant_set")
    p_quality.add_argument("--prepared", default="data/prepared")
    p_quality.add_argument("--out", default=None, help="optional per-document parquet output path")
    p_quality.set_defaults(func=_cmd_quality)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
