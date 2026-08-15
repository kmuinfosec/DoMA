"""CASIMIR paper-revision dataset (HuggingFace ``taln-ls2n/CASIMIR``) -> forum-level docs_df.

The whole pipeline is reproducible from HuggingFace alone: restore the body text from the
sentence pairs, derive per-forum signals, assemble documents. Duplicate-forum removal is an
optional extra branch: ``casimir_raw_*`` keeps every forum as downloaded, while
``casimir_dedup_*`` applies the policy below.

Policy of the dedup branch (**family = forum, never merged**)
-----------------------------------------
A paper resubmitted under several forums (workshop plus main conference, or across splits) is
*dropped*, not merged: only the most recent forum survives and every version of the other forums
is removed. Merging would make the duplicate decision itself the ground-truth label, so a wrong
decision would silently corrupt the evaluation.

Three independent exact-match style signals; any one of them marks two forums as the same paper:
  1. body char-20 shingle Jaccard >= JACCARD_TAU (MinHashLSH shortlist, exact Jaccard decides)
  2. exact title match after squashing - catches resubmissions whose body grew a lot, where
     signal 1 is diluted by the larger union
  3. exact OpenReview paperhash match - catches resubmissions that also changed the title
Hand-curated merge lists are deliberately not used (not reproducible).

Signal 1 filters out shingles shared by many documents (venue boilerplate); without that filter
"Formatting Instructions for ICLR 2018" links unrelated papers.

Within a forum, versions are ordered by submission time (``cdate``, falling back to ``tcdate``)
and numbered 1..n. v1 is the original (confidential DB), the latest revision is the leak query.
``mapping.references`` is ordered newest-first, so it is used to link versions to forums and to
pick each forum's representative text for duplicate detection, never to order versions within a
forum (that ordering comes from the version timestamps).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from doma.ids import build_doc_id, build_family_id, make_revision_doc_key
from doma.io import ensure_dir, save_parquet, save_prepared_set
from doma.schemas import normalize_docs_df

# HuggingFace split names (all of them are used).
CASIMIR_SPLITS = ("train", "validation", "test")

# ── duplicate-forum constants ──────────────────────────────────────────────
# A shingle is a 20-character slice of the text with all whitespace and punctuation removed.
# Word n-grams do not work here: CASIMIR bodies come from PDF extraction and contain stray spaces
# inside words ("Q UATERNION R ECURRENT"), which shifts every word shingle of such a document.
SHINGLE_K = 20
# Duplicate-decision threshold on the exact Jaccard.
JACCARD_TAU = 0.25
# LSH shortlist threshold. Lower than the decision threshold so borderline pairs still become
# candidates; the exact Jaccard makes the final call.
LSH_CANDIDATE_TAU = 0.05
# MinHash permutations (shortlisting only - precision comes from the exact Jaccard recomputation).
LSH_NUM_PERM = 128
# Shingles appearing in at least this fraction of documents are venue boilerplate, not content.
BOILERPLATE_MAX_DF = 0.005
# Sample size for the document-frequency estimate (counting all documents is expensive).
BOILERPLATE_SAMPLE = 2000
# Smallest cutoff for which the df estimate is trusted: below it, boilerplate filtering is skipped
# entirely. Without the guard a small sample pushes the cutoff down to 2, which would erase the
# very body text a duplicate pair shares.
BOILERPLATE_MIN_CUTOFF = 5

# ── HuggingFace loaders (all splits concatenated) ──────────────────────────


# Load the ``article_pairs`` config (sentence pairs used to restore body text).
def load_article_pairs_df(splits: tuple[str, ...] = CASIMIR_SPLITS) -> pd.DataFrame:
    from datasets import load_dataset   # lazy import: the pure logic below is testable without it

    ds = load_dataset("taln-ls2n/CASIMIR", "article_pairs")
    return pd.concat([ds[s].to_pandas() for s in splits], ignore_index=True)


# Load the ``mapping`` config (id_forum -> references[version_id]), tagged with its split.
def load_mapping_df(splits: tuple[str, ...] = CASIMIR_SPLITS) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("taln-ls2n/CASIMIR", "mapping")
    return pd.concat([ds[s].to_pandas().assign(split=s) for s in splits], ignore_index=True)


# Load the ``metadata`` config (id = version_id, forum, cdate/tcdate, content JSON with paperhash).
def load_metadata_df() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("taln-ls2n/CASIMIR", "metadata")
    return ds["train"].to_pandas()


# ── parsing helpers ────────────────────────────────────────────────────────


# Missing-value test; ambiguous values such as numpy arrays are treated as present.
def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (ValueError, TypeError):
        return False


# Collapse all whitespace and invisible characters into single spaces.
def _clean_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return " ".join(str(value).split()).strip()


# Coerce a list / ndarray / JSON or python list literal into a list of strings.
# Handling ndarray matters: without it the forum grouping via ``references`` silently fails.
def _parse_listlike(value: object) -> list[str]:
    if isinstance(value, (list, np.ndarray)):
        return [_clean_text(x) for x in value if _clean_text(x)]
    if _is_missing(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(text)
                if isinstance(parsed, list):
                    return [_clean_text(x) for x in parsed if _clean_text(x)]
            except Exception:
                pass
    return []


# Extract the OpenReview paperhash from a content field (JSON string or dict); "" when absent.
def _paperhash(content: object) -> str:
    if isinstance(content, dict):
        data = content
    else:
        try:
            data = json.loads(content) if isinstance(content, str) else {}
        except Exception:
            data = {}
    return _clean_text(data.get("paperhash")) if isinstance(data, dict) else ""


# Sentence type normalization to title / abstract / p.
_SENTENCE_TYPE = {
    "title": "title", "article-title": "title",
    "abstract": "abstract", "paragraph": "p", "p": "p",
}


# Canonical sentence type; unknown types keep their lowercase original form.
def _canonical_sentence_type(value: object) -> str:
    key = _clean_text(value).lower()
    return _SENTENCE_TYPE.get(key, key)


# Stable short hash, used as a surrogate key when sentence_id is missing.
def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


# Safe column access for article_pairs, whose optional columns may be absent.
def _col(df: pd.DataFrame, name: str, default: object) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([default] * len(df), index=df.index)


# ── restore version text from the sentence pairs ───────────────────────────


# Flatten one side (1|2) of article_pairs into a per-version sentence table (title/abstract/p only).
def _build_side_sentence_frame(pairs_df: pd.DataFrame, side: int) -> pd.DataFrame:
    sub = pd.DataFrame({
        "version_id": _col(pairs_df, f"id_version_{side}", "").map(_clean_text),
        "sentence_id": _col(pairs_df, f"id-sentence-{side}", "").map(_clean_text),
        "text": _col(pairs_df, f"text-sentence-{side}", "").map(_clean_text),
        "sentence_type": _col(pairs_df, f"type-sentence-{side}", "").map(_canonical_sentence_type),
        "page": _col(pairs_df, f"page-sentence-{side}", pd.NA),
        "num_section": _col(pairs_df, f"num_section-sentence-{side}", pd.NA),
        "num_paragraph": _col(pairs_df, f"num_paragraph-sentence-{side}", pd.NA),
        "num_sentence": _col(pairs_df, f"num_sentence-sentence-{side}", pd.NA),
        "pair_index": _col(pairs_df, "sentence-pair-index", pd.NA),
        "side": side,
    })
    sub = sub[
        (sub["version_id"] != "")
        & (sub["text"] != "")
        & (sub["sentence_type"].isin({"title", "abstract", "p"}))
    ].copy()
    # Drop lone "Abstract" header lines; the header is re-added during composition.
    sub = sub[~((sub["sentence_type"] == "abstract") & (sub["text"].str.lower() == "abstract"))].copy()

    missing = sub["sentence_id"] == ""
    sub.loc[missing, "sentence_id"] = sub[missing].apply(
        lambda r: f"{r['version_id']}::{r['page']}::{r['num_section']}::"
                  f"{r['num_paragraph']}::{r['num_sentence']}::{_stable_hash(r['text'])}",
        axis=1,
    )
    for col in ("page", "num_section", "num_paragraph", "num_sentence", "pair_index"):
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(10 ** 15)
    return sub


# Merge both sides and de-duplicate on (version_id, sentence_id).
def build_version_sentence_df(pairs_df: pd.DataFrame) -> pd.DataFrame:
    sent = pd.concat(
        [_build_side_sentence_frame(pairs_df, 1), _build_side_sentence_frame(pairs_df, 2)],
        ignore_index=True,
    )
    sort_cols = ["version_id", "page", "num_section", "num_paragraph",
                 "num_sentence", "pair_index", "side", "sentence_id"]
    sent = sent.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return sent.drop_duplicates(subset=["version_id", "sentence_id"], keep="first").reset_index(drop=True)


# Compose one version's sentences, in document order, into its body text.
def _compose_version_text(group: pd.DataFrame) -> str:
    group = group.sort_values(
        ["page", "num_section", "num_paragraph", "num_sentence", "pair_index", "side"], kind="stable")
    parts: list[str] = []
    prev_type: str | None = None
    for row in group.itertuples(index=False):
        text = _clean_text(row.text)
        if not text:
            continue
        if row.sentence_type == "title":
            if not parts or parts[-1] != text:
                parts += [text, ""]
            prev_type = "title"
        elif row.sentence_type == "abstract":
            if prev_type != "abstract":
                parts.append("Abstract")
            parts += [text, ""]
            prev_type = "abstract"
        else:
            parts += [text, ""]
            prev_type = "p"
    return "\n".join(parts).strip()


# article_pairs -> restored body text per version (version_id, text, num_sent_rows).
def build_version_text_df(pairs_df: pd.DataFrame) -> pd.DataFrame:
    sent = build_version_sentence_df(pairs_df)
    rows = []
    for version_id, group in sent.groupby("version_id", sort=False):
        text = _compose_version_text(group)
        if text:
            rows.append({"version_id": version_id, "text": text, "num_sent_rows": len(group)})
    return pd.DataFrame(rows, columns=["version_id", "text", "num_sent_rows"])


# ── link versions to forums and timestamps ─────────────────────────────────


# version_id -> forum_id fallback table derived from mapping.references
# (the forum id itself is also a candidate initial version).
def build_mapping_version_forum_df(mapping_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in mapping_df.itertuples(index=False):
        forum_id = _clean_text(row.id_forum)
        if forum_id:
            rows.append({"version_id": forum_id, "forum_from_mapping": forum_id})
        for version_id in _parse_listlike(row.references):
            if version_id:
                rows.append({"version_id": version_id, "forum_from_mapping": forum_id})
    out = pd.DataFrame(rows, columns=["version_id", "forum_from_mapping"])
    return out.drop_duplicates(subset=["version_id"], keep="first").reset_index(drop=True)


# Attach forum_id and version_ts (cdate, falling back to tcdate) to the restored versions.
# forum: metadata.forum first, then mapping.references, finally the version_id itself.
def build_version_meta_df(
    version_text_df: pd.DataFrame, mapping_df: pd.DataFrame, metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    meta = metadata_df.copy()
    meta["version_id"] = meta["id"].map(_clean_text)
    meta["forum_from_meta"] = meta["forum"].map(_clean_text)
    ts = pd.to_numeric(meta["cdate"], errors="coerce") if "cdate" in meta else pd.Series(np.nan, index=meta.index)
    if "tcdate" in meta.columns:
        ts = ts.fillna(pd.to_numeric(meta["tcdate"], errors="coerce"))
    meta["version_ts"] = ts

    link = build_mapping_version_forum_df(mapping_df)
    out = (version_text_df
           .merge(meta[["version_id", "forum_from_meta", "version_ts"]], on="version_id", how="left")
           .merge(link, on="version_id", how="left"))

    out["forum_id"] = out["forum_from_meta"]
    for fallback in ("forum_from_mapping", "version_id"):
        blank = out["forum_id"].isna() | (out["forum_id"] == "")
        out.loc[blank, "forum_id"] = out.loc[blank, fallback]
    return out[["version_id", "text", "forum_id", "version_ts", "num_sent_rows"]]


# ── duplicate-forum detection ──────────────────────────────────────────────


# Lowercase alphanumerics only - invariant to inserted/removed spaces, hyphens, case, punctuation.
# NFKD comes first because PDF extractions contain ligatures and accents: without decomposition
# [^a-z0-9] eats them whole and "Efficient" (with an fi ligature) becomes "efcient".
def squash(text: str) -> str:
    ascii_only = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_only.lower())


# Text -> set of k-character shingles, the unit of duplicate comparison.
def shingles(text: str, k: int = SHINGLE_K) -> set[str]:
    s = squash(text)
    if len(s) < k:
        return {s} if s else set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


# Exact Jaccard of two shingle sets (0.0 when the union is empty).
def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# Shingles shared by many documents = venue boilerplate rather than paper content.
# Document frequency is estimated on a deterministic sample (forums in sorted order).
def boilerplate_shingles(
    forum_text: dict[str, str],
    k: int = SHINGLE_K,
    max_df: float = BOILERPLATE_MAX_DF,
    sample_size: int = BOILERPLATE_SAMPLE,
) -> set[str]:
    sample = sorted(forum_text)[:sample_size]
    cutoff = max_df * len(sample)
    if cutoff < BOILERPLATE_MIN_CUTOFF:      # sample too small for a meaningful df estimate
        return set()
    counts: Counter[str] = Counter()
    for f in sample:
        counts.update(shingles(forum_text[f], k))
    return {s for s, c in counts.items() if c >= cutoff}


# forum -> comparison shingle set with boilerplate removed. Computed once and reused.
def build_forum_shingles(
    forum_text: dict[str, str],
    k: int = SHINGLE_K,
    max_df: float = BOILERPLATE_MAX_DF,
    sample_size: int = BOILERPLATE_SAMPLE,
) -> dict[str, set[str]]:
    boiler = boilerplate_shingles(forum_text, k, max_df, sample_size)
    return {f: shingles(text, k) - boiler for f, text in forum_text.items()}


# Forum pairs judged duplicates [(forum_a, forum_b, jaccard)].
# MinHashLSH only shortlists candidates (avoiding an O(N^2) scan); the exact Jaccard decides.
def near_duplicate_pairs(
    forum_shingles: dict[str, set[str]],
    tau: float = JACCARD_TAU,
    num_perm: int = LSH_NUM_PERM,
    candidate_tau: float = LSH_CANDIDATE_TAU,
) -> list[tuple[str, str, float]]:
    from datasketch import MinHash, MinHashLSH   # lazy import: pure logic is testable without it

    lsh = MinHashLSH(threshold=candidate_tau, num_perm=num_perm)
    signatures = {}
    for f, sh in forum_shingles.items():
        m = MinHash(num_perm=num_perm)
        if sh:
            m.update_batch([s.encode("utf8") for s in sh])
        signatures[f] = m
        lsh.insert(f, m)

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, float]] = []
    for f in forum_shingles:
        for other in lsh.query(signatures[f]):
            if other == f:
                continue
            key = (f, other) if f < other else (other, f)
            if key in seen:
                continue
            seen.add(key)
            score = jaccard(forum_shingles[key[0]], forum_shingles[key[1]])
            if score >= tau:
                pairs.append((key[0], key[1], score))
    return pairs


# First lines that are not titles - formatting boilerplate PDF extraction emits instead.
# Without this filter every document starting with "Anonymous authors" collapses into one paper.
TITLE_BOILER = ("anonymousauthor", "paperunderdoubleblindreview", "underreviewasaconferencepaper",
                "underreviewattheiclr", "formattinginstructions", "editors", "abstract",
                "conferencepaperaticlr")
# Squashed first lines shorter than this are treated as failed title extraction.
TITLE_MIN_LEN = 25


# Title key of a document (squashed first non-empty line); None when extraction failed.
# Used for exact comparison only - a similarity match would link "Universal Attacks on X" to
# "Universal Adversarial Attack", which are different papers.
def title_key(text: str) -> str | None:
    for line in str(text).split("\n"):
        if not line.strip():
            continue
        key = squash(line)
        if len(key) < TITLE_MIN_LEN or any(b in key for b in TITLE_BOILER):
            return None
        return key
    return None


# Union-Find with path compression; a connected component is one group of duplicate forums.
class UnionFind:
    def __init__(self, items) -> None:
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# Keep only the most recent forum of each duplicate group and drop the rest (no merging).
# Signals: body Jaccard >= tau, exact title match, exact paperhash match.
# Returns (forums to keep, drop log).
def resolve_duplicate_forums(
    forum_text: dict[str, str],
    forum_paperhash: dict[str, str],
    forum_date: dict[str, object],
    tau: float = JACCARD_TAU,
    k: int = SHINGLE_K,
    max_df: float = BOILERPLATE_MAX_DF,
) -> tuple[set[str], pd.DataFrame]:
    forums = list(forum_text)
    uf = UnionFind(forums)
    forum_shingles = build_forum_shingles(forum_text, k, max_df)
    titles = {f: title_key(t) for f, t in forum_text.items()}

    # Signal 1: body char-shingle Jaccard.
    for a, b, _score in near_duplicate_pairs(forum_shingles, tau=tau):
        uf.union(a, b)

    # Signal 2: exact title match (a heavily expanded resubmission dilutes the Jaccard).
    # Signal 3: identical paperhash (a resubmission that also renamed the paper).
    for group in (titles, forum_paperhash):
        members_by_key: dict[str, list[str]] = defaultdict(list)
        for f in forums:
            key = group.get(f)
            if key:
                members_by_key[key].append(f)
        for members in members_by_key.values():
            for other in members[1:]:
                uf.union(members[0], other)

    # Group representative = the most recent forum; missing dates sort last.
    SMALL = -(1 << 62)

    def newest_key(f: str) -> tuple:
        d = forum_date.get(f)
        return (SMALL if d is None else d, f)

    groups: dict[str, list[str]] = defaultdict(list)
    for f in forums:
        groups[uf.find(f)].append(f)

    kept: set[str] = set()
    records = []
    for members in groups.values():
        rep = max(members, key=newest_key)
        kept.add(rep)
        if len(members) == 1:
            continue
        rep_shingles = forum_shingles[rep]
        rep_hash = forum_paperhash.get(rep, "")
        for f in members:
            if f == rep:
                continue
            records.append({
                "dropped_forum_id": f,
                "kept_forum_id": rep,
                "group_size": len(members),
                "jaccard_to_kept": jaccard(forum_shingles[f], rep_shingles),
                "same_title_as_kept": bool(titles[rep] and titles[f] == titles[rep]),
                "same_paperhash_as_kept": bool(rep_hash and forum_paperhash.get(f, "") == rep_hash),
            })
    dropped_df = pd.DataFrame(
        records,
        columns=["dropped_forum_id", "kept_forum_id", "group_size", "jaccard_to_kept",
                 "same_title_as_kept", "same_paperhash_as_kept"],
    ).sort_values(["kept_forum_id", "dropped_forum_id"]).reset_index(drop=True)
    return kept, dropped_df


# Derive the per-forum duplicate signals.
# Returns (forum_text, forum_date, forum_paperhash, forum_split).
def build_forum_signals(
    version_text_df: pd.DataFrame, mapping_df: pd.DataFrame, metadata_df: pd.DataFrame,
) -> tuple[dict, dict, dict, dict]:
    text_by_version = dict(zip(version_text_df["version_id"].map(_clean_text), version_text_df["text"]))
    # A forum is represented by the first version with text in ``references`` (newest-first order),
    # i.e. its most complete version, since duplicate detection compares one text per forum.
    forum_text: dict[str, str] = {}
    for forum, refs in zip(mapping_df["id_forum"], mapping_df["references"]):
        forum = _clean_text(forum)
        if not forum or forum in forum_text:
            continue
        for version_id in _parse_listlike(refs):
            if version_id in text_by_version:
                forum_text[forum] = str(text_by_version[version_id])
                break

    meta = metadata_df.copy()
    meta["fid"] = meta["forum"].map(_clean_text)

    # A forum's date is the earliest tcdate among its versions (its original submission time),
    # falling back to cdate then pdate when tcdate is absent.
    def _forum_date(group: pd.DataFrame) -> object:
        for col in ("tcdate", "cdate", "pdate"):
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(values):
                    return int(values.min())
        return None

    forum_date = meta[meta["fid"] != ""].groupby("fid").apply(_forum_date, include_groups=False).to_dict()

    meta["ph"] = meta["content"].map(_paperhash)
    ph = meta[(meta["fid"] != "") & (meta["ph"] != "")].drop_duplicates("fid")
    forum_paperhash = dict(zip(ph["fid"], ph["ph"]))

    forum_split = dict(zip(mapping_df["id_forum"].map(_clean_text), mapping_df.get("split")))
    return forum_text, forum_date, forum_paperhash, forum_split


# ── forum -> docs_df (pure, no HuggingFace needed) ─────────────────────────


# Assemble documents, one family per forum. Versions of dropped forums disappear entirely, and
# version_index 1..n is assigned within each surviving forum by ascending timestamp.
# kept_forums=None keeps every forum — the primary casimir_raw branch.
def _assemble_docs_df(
    version_meta_df: pd.DataFrame,
    kept_forums: set[str] | None,
    split_of: dict[str, str],
) -> pd.DataFrame:
    df = version_meta_df.copy()
    if kept_forums is not None:
        df = df[df["forum_id"].isin(kept_forums)]
    df = df.sort_values(["forum_id", "version_ts", "version_id"],
                        kind="stable", na_position="last").reset_index(drop=True)
    df["version_index"] = df.groupby("forum_id").cumcount() + 1

    rows = []
    for row in df.itertuples(index=False):
        family_id = build_family_id("casimir", row.forum_id)
        version_index = int(row.version_index)
        rows.append({
            "doc_id": build_doc_id(family_id, make_revision_doc_key(version_index)),
            "dataset": "casimir",
            "family_id": family_id,
            "text": row.text,
            "version_index": version_index,
            "variant_type": "original" if version_index == 1 else "revision",
            "source_doc_id": build_doc_id(family_id, "v1") if version_index > 1 else None,
            "meta_json": {
                "forum_id": row.forum_id,
                "raw_version_id": row.version_id,
                "split": split_of.get(row.forum_id),
                "version_ts": None if pd.isna(row.version_ts) else float(row.version_ts),
                # number of sentence rows used in the restoration (audit for suspiciously short docs)
                "num_sent_rows": None if pd.isna(row.num_sent_rows) else int(row.num_sent_rows),
            },
        })
    return normalize_docs_df(pd.DataFrame(rows))


# Full CASIMIR build -> (deduplicated docs, drop log, docs without deduplication).
# The third output feeds casimir_raw sets; the first two are the optional dedup branch and its drop log.
def build_casimir_docs_df(
    splits: tuple[str, ...] = CASIMIR_SPLITS, tau: float = JACCARD_TAU,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs_df = load_article_pairs_df(splits)
    mapping_df = load_mapping_df(splits)
    metadata_df = load_metadata_df()

    version_text_df = build_version_text_df(pairs_df)
    version_meta_df = build_version_meta_df(version_text_df, mapping_df, metadata_df)

    forum_text, forum_date, forum_paperhash, forum_split = build_forum_signals(
        version_text_df, mapping_df, metadata_df)
    kept, dropped_df = resolve_duplicate_forums(forum_text, forum_paperhash, forum_date, tau=tau)
    return (_assemble_docs_df(version_meta_df, kept, forum_split),
            dropped_df,
            _assemble_docs_df(version_meta_df, None, forum_split))


# ── prepared set assembly ──────────────────────────────────────────────────


# docs_df -> (v1_only, revisions = everything after v1, latest_only).
# Forums without a later version are excluded: they carry no revision to detect.
def split_casimir_prepared_sets(
    docs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_version = docs_df.groupby("family_id")["version_index"].max()
    eligible = max_version[max_version >= 2].index
    work = docs_df[docs_df["family_id"].isin(eligible)].copy()

    v1 = work[work["version_index"] == 1].reset_index(drop=True)
    revisions = work[work["version_index"] > 1].reset_index(drop=True)
    latest = (revisions.sort_values(["family_id", "version_index"], kind="stable")
              .groupby("family_id", as_index=False, sort=False).tail(1).reset_index(drop=True))
    return v1, revisions, latest


# Write both branches (dedup and raw) plus the drop log. Returns per-set document counts.
# Naming: ``casimir_{raw|dedup}_{v1_only|revisions|latest_only}``.
def build_and_save_casimir_prepared_sets(
    prepared_dir: str | Path,
    splits: tuple[str, ...] = CASIMIR_SPLITS,
    tau: float = JACCARD_TAU,
) -> dict[str, int]:
    docs_df, dropped_df, raw_docs_df = build_casimir_docs_df(splits=splits, tau=tau)

    counts: dict[str, int] = {}
    for prefix, df in (("dedup", docs_df), ("raw", raw_docs_df)):
        v1, revisions, latest = split_casimir_prepared_sets(df)
        for suffix, part in (("v1_only", v1), ("revisions", revisions), ("latest_only", latest)):
            name = f"casimir_{prefix}_{suffix}"
            save_prepared_set(part, prepared_dir, name)
            counts[name] = len(part)

    # Drop log; not a prepared set, stored separately.
    save_parquet(dropped_df, Path(ensure_dir(prepared_dir)) / "casimir_dedup_dropped_forums.parquet")
    counts["dropped_forums"] = len(dropped_df)
    return counts
