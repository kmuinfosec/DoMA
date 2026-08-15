"""File IO helpers (parquet / json / yaml) and prepared-set IO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from doma.schemas import normalize_docs_df


# Create the directory if needed and return it as a Path.
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Write a JSON file (unicode preserved, non-serializable values fall back to str).
def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# Read a YAML file (used for experiment configs under configs/).
def load_yaml(path: str | Path) -> Any:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Write a DataFrame to parquet (index excluded).
def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# Normalize a docs_df and store it as '<set_name>.parquet'.
def save_prepared_set(df: pd.DataFrame, prepared_dir: str | Path, set_name: str) -> Path:
    prepared_dir = ensure_dir(prepared_dir)
    out_df = normalize_docs_df(df)
    path = prepared_dir / f"{set_name}.parquet"
    out_df.to_parquet(path, index=False)
    return path


# Load a prepared set ('<set_name>.parquet'); raises when it does not exist.
def load_prepared_set(prepared_dir: str | Path, set_name: str) -> pd.DataFrame:
    path = Path(prepared_dir) / f"{set_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"prepared set not found: {path}")
    return pd.read_parquet(path)


# Concatenate several prepared sets into one docs_df, dropping duplicate doc_id.
def concat_prepared_sets(prepared_dir: str | Path, set_names: list[str]) -> pd.DataFrame:
    if not set_names:
        raise ValueError("set_names is empty.")
    dfs = [load_prepared_set(prepared_dir, name) for name in set_names]
    out = pd.concat(dfs, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    return normalize_docs_df(out)
