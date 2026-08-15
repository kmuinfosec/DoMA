"""PAR3 parallel-translation corpus (par3.pkl) -> document-level canonical docs_df.

One work = its paragraphs joined by newlines into a single document.
    par3_gt       : the Google-MT translation (registered as confidential, ``original``)
    par3_human    : one document per human translator (the leak, ``human_translation``)
    par3_human_t1 : translator_1 only
family_id = work_id, so a work's machine translation and every human translation share a family.

Two variants of each set are written: volumes merged (``par3_gt``) and volumes kept apart
(``par3_gt_split``), where one book volume is one family.

PAR3 ids are bespoke: work_id is used directly as family_id, without a "par3:" prefix.
"""
from __future__ import annotations

import pickle
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from doma.io import save_prepared_set
from doma.schemas import normalize_docs_df

# Trailing _xx language code of a work_id.
_LANG_RE = re.compile(r"_([a-z]{2})$")
# Infix _<volume number>_ marking a multi-volume work.
_VOL_RE = re.compile(r"_(\d+)_")
_LANG_NAME = {
    "cs": "Czech", "de": "German", "es": "Spanish", "fa": "Persian",
    "fr": "French", "hu": "Hungarian", "it": "Italian", "ja": "Japanese",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "sv": "Swedish", "ta": "Tamil", "zh": "Chinese",
}


# Split a work_id into (title slug, language code).
def _split_lang(work_id: str) -> tuple[str, str]:
    m = _LANG_RE.search(work_id)
    return (work_id[: m.start()], m.group(1)) if m else (work_id, "unknown")


# Join paragraphs with newlines, dropping empty ones.
def _join(paras: list[str]) -> str:
    return "\n".join(s for p in paras if (s := (p or "").strip()))


# Concatenate the volumes of a multi-volume work, in volume order, into one work.
# Single-volume and non-volumed works pass through unchanged.
def _merge_volumes(data: dict) -> dict:
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    singles: list[str] = []
    for work_id in sorted(data):
        m = _VOL_RE.search(work_id)
        if m:
            groups[work_id[: m.start()]].append((int(m.group(1)), work_id))
        else:
            singles.append(work_id)

    result: dict[str, dict] = {work_id: data[work_id] for work_id in singles}
    for base, items in groups.items():
        if len(items) == 1:
            result[items[0][1]] = data[items[0][1]]
            continue
        items.sort()
        _, lang = _split_lang(items[0][1])
        translators = sorted(data[items[0][1]]["translator_data"].keys())
        gt, source = [], []
        translator_paras: dict[str, list] = defaultdict(list)
        for _, work_id in items:
            work = data[work_id]
            gt += work["gt_paras"]
            source += work["source_paras"]
            for tk in translators:
                translator_paras[tk] += work["translator_data"][tk]["translator_paras"]
        result[f"{base}_{lang}"] = {
            "gt_paras": gt,
            "source_paras": source,
            "translator_data": {tk: {"translator_paras": translator_paras[tk]} for tk in translators},
        }
    return result


# par3.pkl -> (machine-translation docs_df, human-translation docs_df).
# merge_volumes=False keeps each volume as its own work.
def build_par3_frames(pkl_path: str | Path, merge_volumes: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    works = _merge_volumes(data) if merge_volumes else data

    gt_rows: list[dict] = []
    human_rows: list[dict] = []
    for work_id, work in sorted(works.items()):
        _, lang = _split_lang(work_id)
        translators = sorted(work["translator_data"].keys())
        meta = {
            "lang": lang,
            "lang_name": _LANG_NAME.get(lang, lang),
            "n_paragraphs": len(work["gt_paras"]),
            "n_translators": len(translators),
        }
        gt_rows.append({
            "doc_id": f"{work_id}__gt",
            "dataset": "par3",
            "family_id": work_id,
            "text": _join(work["gt_paras"]),
            "version_index": 0,
            "variant_type": "original",
            "source_doc_id": None,
            "meta_json": {**meta, "role": "machine_translation"},
        })
        for version_index, tk in enumerate(translators, start=1):
            text = _join(work["translator_data"][tk]["translator_paras"])
            if not text:
                continue
            human_rows.append({
                "doc_id": f"{work_id}__{tk}",
                "dataset": "par3",
                "family_id": work_id,
                "text": text,
                "version_index": version_index,
                "variant_type": "human_translation",
                "variant_level": tk,
                "source_doc_id": f"{work_id}__gt",
                "meta_json": {**meta, "translator": tk, "role": "human_translation"},
            })
    return normalize_docs_df(pd.DataFrame(gt_rows)), normalize_docs_df(pd.DataFrame(human_rows))


# Write both variants: volumes merged (par3_gt/human/human_t1) and volumes split (par3_*_split).
def build_and_save_par3_prepared_sets(pkl_path: str | Path, prepared_dir: str | Path) -> None:
    for suffix, merge in (("", True), ("_split", False)):
        gt_df, human_df = build_par3_frames(pkl_path, merge_volumes=merge)
        human_t1 = human_df[human_df["variant_level"] == "translator_1"].reset_index(drop=True)
        save_prepared_set(gt_df, prepared_dir, f"par3_gt{suffix}")
        save_prepared_set(human_df, prepared_dir, f"par3_human{suffix}")
        save_prepared_set(human_t1, prepared_dir, f"par3_human_t1{suffix}")
