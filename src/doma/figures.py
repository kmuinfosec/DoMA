"""The ROC figure.

Curves are rebuilt from each run's ``per_query_eval.parquet`` — the same labels and raw confidences
that produced ``metrics.json`` — so drawing the figure re-runs nothing. A method with no run on a
panel's dataset is left out; a panel with no runs at all says so in its title.

    python -m doma.cli figure --out artifacts/figures/roc.pdf
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from doma.report import PAPER_ROWS, load_all_metrics, select_axis

# Panels of the figure, left to right: (dataset key, panel title).
FIGURE_PANELS: list[tuple[str, str]] = [
    ("par3", "PAR3"),
    ("casimir_raw", "CASIMIR"),
]

# Line style per row of PAPER_ROWS.
LINE_STYLES: dict[str, dict] = {
    "DoMA (Qwen3 0.6B)":  {"color": "#009e8e", "linestyle": "-",   "linewidth": 1.8},
    "DoMA (Granite 97M)": {"color": "#e6a817", "linestyle": "--",  "linewidth": 1.5},
    "DoMA (BGE 109M)":    {"color": "#1f6fd0", "linestyle": "-",   "linewidth": 1.5},
    "BM25":               {"color": "#222222", "linestyle": "-.",  "linewidth": 1.3},
    "Fuzzy hash":         {"color": "#d94040", "linestyle": (0, (5, 2)), "linewidth": 1.3},
}


# Directory of the single run behind one (row, dataset) cell, or None when it was never run.
# Several matching runs raise, as in report.paper_table.
def _run_dir(artifacts_dir, metrics_df, method, tag_substring, dataset,
             variant_sets, include_original_as_positive) -> Path | None:
    sub = metrics_df[metrics_df["method"] == method]
    if tag_substring:
        sub = sub[sub["run_tag"].str.contains(tag_substring, case=False, regex=False, na=False)]
    cell = select_axis(sub, dataset, variant_sets, include_original_as_positive)
    runs = cell[["run_ident", "run_tag"]].drop_duplicates()
    if runs.empty:
        return None
    if len(runs) > 1:
        listed = ", ".join(f"{r.run_ident}/{r.run_tag}" for r in runs.itertuples())
        raise ValueError(f"{method}/{tag_substring} x {dataset} matches {len(runs)} runs: {listed}")
    row = runs.iloc[0]
    return Path(artifacts_dir) / "runs" / row["run_ident"] / row["run_tag"]


# (labels, scores) of one run. A query with no match scores 0, exactly as in evaluate_run.
def _labels_and_scores(run_dir: Path) -> tuple[pd.Series, pd.Series] | None:
    path = run_dir / "per_query_eval.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or df["is_positive"].nunique() < 2:
        return None
    return df["is_positive"].astype(int), df["confidence"].fillna(0.0)


# Draw the ROC panels and write them to out_path (the suffix decides the format).
# Returns the AUROC of every curve drawn, keyed by (panel, method label).
def roc_figure(artifacts_dir: str | Path = "artifacts",
               out_path: str | Path = "artifacts/figures/roc.pdf",
               panels: list[tuple[str, str]] = FIGURE_PANELS,
               rows_spec: list[tuple[str, str, str | None]] = PAPER_ROWS,
               include_original_as_positive: bool = False,
               variant_sets: dict[str, tuple[str, ...]] | None = None) -> dict[tuple[str, str], float]:
    import matplotlib.pyplot as plt

    variant_sets = variant_sets or {}
    metrics_df = load_all_metrics(artifacts_dir)
    aurocs: dict[tuple[str, str], float] = {}

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.6), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (dataset, title) in zip(axes, panels):
        drawn = 0
        for row_label, method, tag_substring in rows_spec:
            run_dir = _run_dir(artifacts_dir, metrics_df, method, tag_substring, dataset,
                               variant_sets.get(dataset), include_original_as_positive)
            data = _labels_and_scores(run_dir) if run_dir else None
            if data is None:
                continue
            labels, scores = data
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = float(roc_auc_score(labels, scores))
            aurocs[(dataset, row_label)] = auroc
            ax.plot(fpr, tpr, label=f"{row_label} ({auroc:.3f})", **LINE_STYLES.get(row_label, {}))
            drawn += 1

        ax.plot([0, 1], [0, 1], color="0.75", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_title(title if drawn else f"{title} — no runs found")
        if drawn:
            ax.legend(loc="lower right", fontsize=7, frameon=False)
    axes[0].set_ylabel("True positive rate")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    # A .pdf request also gets a .png, so the figure can be eyeballed without a viewer.
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return aurocs
