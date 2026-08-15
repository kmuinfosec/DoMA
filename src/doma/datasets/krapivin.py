"""Krapivin scientific-paper corpus -> canonical docs_df.

Source file: ``KEYWORD_DATA/krapivin_test.json`` from https://github.com/EruM16/Attention-Seeker.

Source format: JSON array or JSONL, one record per paper
``{name, title, abstract, fulltext, keywords}``.
    text      = title\\n\\nAbstract\\n{abstract}\\n\\n{fulltext}, whitespace collapsed
    family_id = <dataset>:<name>, doc_id = <family_id>:orig
Keywords are kept in meta_json only; they are never part of the indexed text.

Redistributions with a different record layout exist (one carries ``{file_name, text, keyphrases}``
and no title/abstract separation); they are rejected rather than read as zero records.

The same reader handles any SemEval-style keyphrase JSONL, hence the ``dataset`` argument.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from doma.ids import build_doc_id, build_family_id
from doma.io import save_prepared_set
from doma.schemas import normalize_docs_df


# Collapse every run of whitespace (including nbsp and CR) into a single space.
def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace(" ", " ").replace("\r", " ")
    return " ".join(text.split()).strip()


# Assemble one record into the document body: title, abstract (with header), full text.
def _record_to_text(rec: dict) -> str:
    title, abstract, fulltext = _clean(rec.get("title")), _clean(rec.get("abstract")), _clean(rec.get("fulltext"))
    parts: list[str] = []
    if title:
        parts += [title, ""]
    if abstract:
        parts += ["Abstract", abstract, ""]
    if fulltext:
        parts += [fulltext]
    return "\n".join(parts).strip()


# Read the source file as a record list (accepts both a JSON array and JSONL).
def _load_records(src_path: str | Path) -> list[dict]:
    txt = Path(src_path).read_text(encoding="utf-8").strip()
    if not txt:
        return []
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


# Convert the raw keyphrase corpus into a canonical docs_df.
def build_krapivin_docs_df(src_path: str | Path, dataset: str = "krapivin") -> pd.DataFrame:
    rows: list[dict] = []
    for rec in _load_records(src_path):
        name = _clean(rec.get("name"))
        text = _record_to_text(rec)
        if not name or not text:
            continue
        family_id = build_family_id(dataset, name)
        rows.append({
            "doc_id": build_doc_id(family_id, "orig"),
            "dataset": dataset,
            "family_id": family_id,
            "text": text,
            "variant_type": "original",
            "source_doc_id": None,
            "meta_json": {"raw_name": name, "title": _clean(rec.get("title")),
                          "keywords": _clean(rec.get("keywords")),
                          "source_path": str(src_path)},
        })
    if not rows:
        records = _load_records(src_path)
        found = sorted(records[0]) if records else []
        raise ValueError(
            f"{src_path}: parsed {len(records)} records but none carried both 'name' and text. "
            f"Expected the keys {{name, title, abstract, fulltext, keywords}}; found {found}.")
    return normalize_docs_df(pd.DataFrame(rows))


# Build and store the original set as '{dataset}_original.parquet'.
def build_and_save_krapivin_original(src_path: str | Path, prepared_dir: str | Path,
                                     dataset: str = "krapivin") -> None:
    save_prepared_set(build_krapivin_docs_df(src_path, dataset), prepared_dir, f"{dataset}_original")
