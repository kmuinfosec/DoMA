"""Family-aware 50/50 confidential split.

Invariant: every document of a family (original, variants, revisions) lands in the same
partition, so no family is ever split across the confidential DB and the benign pool.
The split is seeded and therefore reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from doma.io import (
    concat_prepared_sets,
    ensure_dir,
    load_prepared_set,
    save_parquet,
)

SPLIT_COLUMNS = ["family_id", "partition", "dataset", "seed"]
PARTITIONS = ("confidential", "benign")


# Shuffle the families of original_set and mark `ratio` of them confidential.
# Reuses the stored split file unless overwrite=True; the file name carries only original_set
# and seed, so changing ratio requires overwrite=True to take effect.
# The file is keyed by original_set, not by dataset: two original sets of the same dataset can
# have different family universes, and reusing a stale split across them silently corrupts results.
def make_split(
    prepared_dir: str | Path,
    splits_dir: str | Path,
    dataset: str,
    original_set: str,
    seed: int = 42,
    ratio: float = 0.5,
    overwrite: bool = False,
) -> pd.DataFrame:
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0,1): {ratio}")

    out_path = Path(splits_dir) / f"{original_set}__s{seed}.parquet"
    if out_path.exists() and not overwrite:
        return pd.read_parquet(out_path)

    docs = load_prepared_set(prepared_dir, original_set)
    families = np.array(sorted(docs["family_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(families)

    n_conf = int(round(len(families) * ratio))
    confidential = set(families[:n_conf].tolist())

    split_df = pd.DataFrame({
        "family_id": families,
        "partition": ["confidential" if f in confidential else "benign" for f in families],
        "dataset": dataset,
        "seed": int(seed),
    })[SPLIT_COLUMNS]

    ensure_dir(splits_dir)
    save_parquet(split_df, out_path)
    return split_df


# Keep only the documents whose family belongs to the requested partition.
def apply_split(docs_df: pd.DataFrame, split_df: pd.DataFrame, partition: str) -> pd.DataFrame:
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}: {partition!r}")
    target = set(split_df.loc[split_df["partition"] == partition, "family_id"])
    return docs_df[docs_df["family_id"].isin(target)].reset_index(drop=True)


# Check that every family of a document set exists in the split; raise when strict, warn otherwise.
# A family absent from the split falls into neither partition, so its documents are dropped from
# both the positive and the benign side.
def _assert_family_coverage(docs_df, split_df, set_name, strict=False) -> None:
    missing = set(docs_df["family_id"].unique()) - set(split_df["family_id"].unique())
    if missing:
        msg = f"[splits] {set_name}: {len(missing)} families absent from the split (e.g. {sorted(missing)[:5]})"
        if strict:
            raise ValueError(msg)
        print(f"WARNING {msg}")


# The three document sets of one run.
@dataclass(frozen=True)
class RunQuerySets:
    reference_df: pd.DataFrame   # confidential originals = the registered confidential DB
    positive_df: pd.DataFrame    # variants of confidential documents (+ optionally the originals) = leaks
    benign_df: pd.DataFrame      # benign-side originals and their variants


# Apply the split and assemble reference / positive / benign sets.
# include_original_as_positive=True also screens the confidential originals themselves
# (verbatim leak); default False means positives are variants only.
def build_run_query_sets(
    prepared_dir: str | Path,
    split_df: pd.DataFrame,
    original_set: str,
    variant_sets: list[str],
    include_original_as_positive: bool = False,
    strict: bool = False,
) -> RunQuerySets:
    original_df = load_prepared_set(prepared_dir, original_set)
    _assert_family_coverage(original_df, split_df, original_set, strict=strict)

    if variant_sets:
        variants_df = concat_prepared_sets(prepared_dir, variant_sets)
        _assert_family_coverage(variants_df, split_df, "+".join(variant_sets), strict=strict)
    else:
        variants_df = original_df.iloc[0:0].copy()

    reference_df = apply_split(original_df, split_df, "confidential")

    positive_parts = [apply_split(variants_df, split_df, "confidential")]
    if include_original_as_positive:
        positive_parts.append(apply_split(original_df, split_df, "confidential"))
    positive_df = pd.concat(positive_parts, ignore_index=True).drop_duplicates(subset=["doc_id"])

    benign_df = pd.concat(
        [apply_split(original_df, split_df, "benign"), apply_split(variants_df, split_df, "benign")],
        ignore_index=True,
    ).drop_duplicates(subset=["doc_id"])

    return RunQuerySets(
        reference_df=reference_df.reset_index(drop=True),
        positive_df=positive_df.reset_index(drop=True),
        benign_df=benign_df.reset_index(drop=True),
    )
