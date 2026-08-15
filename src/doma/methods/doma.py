"""DoMA — Document Membership Attribution.

One document becomes exactly one vector, and detection is a cosine nearest-neighbour lookup
against the registered confidential documents. There is no chunk index and no vote aggregation:
the whole document is the unit of both registration and comparison.

How the single vector is built
------------------------------
1. Tokenize the document without truncation.
2. If it fits in the model's maximum token window, encode it directly.
3. Otherwise cut the body into consecutive windows (that maximum length minus the special
   tokens), encode each window, and take the **unweighted mean** of the (un-normalized)
   window embeddings.
4. L2-normalize the result, so inner product against the index equals cosine.

The window length is the model's own maximum token limit, discovered from the loaded model
rather than hard-coded, so the same code path holds for long-context and short-context encoders. Each window leaves room for the special tokens
the encoder re-adds.

``confidence`` is the raw cosine to the best-matching confidential family. It is evaluated by a
threshold sweep, never by a fixed cut-off.

Embeddings are cached **per prepared set**, so runs that differ only in split, variant combination
or query construction reuse the same vectors. torch / sentence_transformers are imported lazily.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from doma.ids import model_slug

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# One loaded model per configuration, reused across sets within a process.
_MODEL_CACHE: dict[str, object] = {}
# The window each model reported at load time. Callers set ``model.max_seq_length`` on the shared
# object, so the native value is kept here rather than read back off the model.
_NATIVE_WINDOW: dict[str, int] = {}


# Load a SentenceTransformer at its native maximum window (lazy import, process-level cache).
# device=None picks cuda when available.
def _get_encoder(model_name: str, device: str | None, dtype: str):
    import torch
    from sentence_transformers import SentenceTransformer

    key = f"{model_name}|{device}|{dtype}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Running 32k attention through the math kernel materializes an N x N matrix and OOMs, so force
    # the memory-efficient / flash SDPA kernels. This is a no-op for short documents.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    model = SentenceTransformer(model_name, device=device,
                                model_kwargs={"torch_dtype": torch_dtype}, trust_remote_code=True)
    if getattr(model, "tokenizer", None) is not None:
        model.tokenizer.truncation_side = "right"   # keep the head if a window ever needs trimming
    model.eval()
    _MODEL_CACHE[key] = model
    _NATIVE_WINDOW[key] = int(model.max_seq_length)
    return model


# The model's own maximum token window, cached on disk so later runs need no model load.
# This is the DoMA pooling window: longer documents are split into windows of this size minus
# the special tokens.
def resolve_max_seq_length(model_name: str, artifacts_dir: str | Path,
                           device: str | None = None, dtype: str = "float32") -> int:
    path = Path(artifacts_dir) / "cache" / "embeddings_doma" / model_slug(model_name) / "max_seq_length.json"
    if path.exists():
        return int(json.loads(path.read_text(encoding="utf-8"))["max_seq_length"])
    _get_encoder(model_name, device, dtype)
    n = _NATIVE_WINDOW[f"{model_name}|{device}|{dtype}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_name": model_name, "max_seq_length": n}, indent=2),
                    encoding="utf-8")
    return n


# Encode documents into one mean-pooled, L2-normalized vector each.
# Returns (embeddings (N,D), number of documents that exceeded one window).
def _mean_pooled_doc_vectors(model, texts: list[str], max_seq_length: int, batch_size: int,
                             desc: str | None = None):
    def _encode(segment_texts, normalize):
        if not segment_texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()), np.float32)
        return model.encode(segment_texts, batch_size=batch_size, normalize_embeddings=normalize,
                            convert_to_numpy=True,
                            show_progress_bar=True).astype(np.float32)

    tok = model.tokenizer
    # Tokenize as ``model.encode`` would (special tokens included); over-window documents are
    # chunked with the special tokens subtracted, since encoding re-adds them per window.
    ids_list = tok(texts, truncation=False, padding=False, add_special_tokens=True,
                   return_attention_mask=False)["input_ids"]
    n_overflow = int(sum(len(ids) > max_seq_length for ids in ids_list))

    window = max(1, max_seq_length - 2)   # room for the special tokens re-added on encode
    segment_texts: list[str] = []
    segment_owner: list[int] = []
    for doc_index, ids in enumerate(ids_list):
        if len(ids) <= max_seq_length:
            segment_texts.append(texts[doc_index])
            segment_owner.append(doc_index)
            continue
        body = ids[1:-1] if len(ids) >= 2 else ids
        for start in range(0, len(body), window):
            segment_texts.append(tok.decode(body[start: start + window], skip_special_tokens=True))
            segment_owner.append(doc_index)

    if desc:
        print(f"[encode] {desc}: {len(texts)} docs -> {len(segment_texts)} segments "
              f"({n_overflow} over one window), batch_size={batch_size}", flush=True)

    # Mean is taken over un-normalized window embeddings, then normalized once at the end.
    segment_emb = _encode(segment_texts, normalize=False)
    out = np.zeros((len(texts), segment_emb.shape[1]), np.float32)
    counts = np.zeros(len(texts))
    for emb, owner in zip(segment_emb, segment_owner):
        out[owner] += emb
        counts[owner] += 1
    counts[counts == 0] = 1
    out /= counts[:, None]
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out, n_overflow


# Encode one whole prepared set and cache it on disk.
# The cache key is the set name (plus model / window / dtype), never the split or variant
# combination, so every run that touches this set reuses the same vectors.
# Returns (embeddings, meta[doc_id, family_id], timing info).
def _encode_set(set_name, model_name, max_seq_length, batch_size, device, dtype,
                artifacts_dir, prepared_dir, force_rebuild=False):
    cache_root = Path(artifacts_dir) / "cache" / "embeddings_doma" / model_slug(model_name)
    cache_root.mkdir(parents=True, exist_ok=True)
    tag = f"{set_name}__L{max_seq_length}__{dtype}"
    npy_path = cache_root / f"{tag}.npy"
    meta_path = cache_root / f"{tag}.parquet"
    info_path = cache_root / f"{tag}.meta.json"

    if not force_rebuild and npy_path.exists() and meta_path.exists() and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return np.load(npy_path), pd.read_parquet(meta_path), {
            "encode_sec": float(info.get("encode_sec", 0.0)),
            "source": "cached",
            "n_overflow": int(info.get("n_overflow", 0)),
        }

    from doma.io import load_prepared_set

    docs_df = load_prepared_set(prepared_dir, set_name)
    model = _get_encoder(model_name, device, dtype)
    model.max_seq_length = int(max_seq_length)
    sub = docs_df[["doc_id", "family_id", "text"]].sort_values("doc_id").reset_index(drop=True)
    texts = [t if str(t).strip() else " " for t in sub["text"].fillna("").astype(str)]

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # release memory reserved by a previous set to limit fragmentation

    started = perf_counter()
    emb, n_overflow = _mean_pooled_doc_vectors(model, texts, max_seq_length, batch_size,
                                               desc=set_name)
    encode_sec = perf_counter() - started

    meta_df = pd.DataFrame({"doc_id": sub["doc_id"], "family_id": sub["family_id"]})
    np.save(npy_path, emb)
    meta_df.to_parquet(meta_path, index=False)
    info_path.write_text(json.dumps({
        "model_name": model_name, "max_seq_length": int(max_seq_length),
        "encode_sec": float(encode_sec), "n_docs": int(len(meta_df)),
        "n_overflow": int(n_overflow)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return emb, meta_df, {"encode_sec": float(encode_sec), "source": "live", "n_overflow": int(n_overflow)}


# Prorate a set's cached encoding time by the share of its documents actually used in this run.
def _prorated_encode_sec(per_set: list[tuple[set, float]], used_ids: set) -> float:
    total = 0.0
    for ids, sec in per_set:
        if ids:
            total += sec * (len(used_ids & ids) / len(ids))
    return total


# Register the confidential document vectors in a FAISS index and search it with the query vectors.
# Returns (matches_df, index_build_sec, search_sec). Vectors must be L2-normalized.
def _register_and_search(ref_emb, ref_doc_ids, ref_families,
                         query_emb, query_doc_ids, query_families, faiss_config, top_k):
    from doma.index.faiss_hnsw import build_faiss_hnsw_index
    from doma.matches import MATCH_COLUMNS, best_match_by_document
    from doma.retrieval.core import search_index

    if len(ref_doc_ids) == 0 or len(query_doc_ids) == 0:
        return pd.DataFrame(columns=MATCH_COLUMNS), 0.0, 0.0

    build_started = perf_counter()
    index = build_faiss_hnsw_index(
        np.asarray(ref_emb, dtype=np.float32),
        pd.DataFrame({"doc_id": ref_doc_ids, "family_id": ref_families}),
        faiss_config)
    index_build_sec = perf_counter() - build_started

    search_started = perf_counter()
    retrieval = search_index(index, query_doc_ids, query_families,
                             np.asarray(query_emb, dtype=np.float32),
                             min(len(ref_doc_ids), top_k))
    matches = best_match_by_document(retrieval)
    search_sec = perf_counter() - search_started
    return matches, index_build_sec, search_sec


# DoMA end to end: (reference documents, query documents) -> (matches_df, timing).
# set_names lists the prepared sets this run is assembled from (original plus variants); each is
# encoded and cached as a whole, then reference/query vectors are selected from them by doc_id.
# max_seq_length=None uses the model's own maximum token window (the DoMA default).
def doma_matches(
    reference_docs_df: pd.DataFrame,
    query_docs_df: pd.DataFrame,
    set_names: list[str],
    prepared_dir: str | Path = "data/prepared",
    model_name: str = DEFAULT_MODEL,
    max_seq_length: int | None = None,
    batch_size: int = 8,
    device: str | None = None,
    dtype: str = "bfloat16",
    artifacts_dir: str | Path = "artifacts",
    faiss_config=None,
    top_k: int = 1,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict]:
    if faiss_config is None:
        from doma.index.faiss_hnsw import FAISSHNSWConfig

        faiss_config = FAISSHNSWConfig()
    if max_seq_length is None:
        max_seq_length = resolve_max_seq_length(model_name, artifacts_dir, device, dtype)

    started = perf_counter()

    # ---- encode each prepared set (cached), then map doc_id -> vector ----
    vec: dict[object, np.ndarray] = {}
    per_set: list[tuple[set, float]] = []
    sources: list[str] = []
    n_overflow_total = 0
    for set_name in set_names:
        emb, meta, info = _encode_set(set_name, model_name, max_seq_length, batch_size, device, dtype,
                                      artifacts_dir, prepared_dir, force_rebuild)
        ids = meta["doc_id"].tolist()
        vec.update(zip(ids, emb))
        per_set.append((set(ids), float(info["encode_sec"])))
        sources.append(info["source"])
        n_overflow_total += int(info["n_overflow"])

    # ---- assemble reference and query matrices ----
    ref_ids = list(reference_docs_df["doc_id"])
    query_ids = list(query_docs_df["doc_id"])
    missing = [d for d in (*ref_ids, *query_ids) if d not in vec]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(ref_ids) + len(query_ids)} documents have no embedding in "
            f"sets {list(set_names)} (e.g. {missing[:5]}); the prepared sets, the split and the "
            f"run configuration are out of sync.")
    ref_family_of = dict(zip(reference_docs_df["doc_id"], reference_docs_df["family_id"]))
    query_family_of = dict(zip(query_docs_df["doc_id"], query_docs_df["family_id"]))
    dim = next(iter(vec.values())).shape[0] if vec else 0
    ref_emb = np.stack([vec[d] for d in ref_ids]) if ref_ids else np.zeros((0, dim), np.float32)
    query_emb = np.stack([vec[d] for d in query_ids]) if query_ids else np.zeros((0, dim), np.float32)

    ref_encode_sec = _prorated_encode_sec(per_set, set(ref_ids))
    query_encode_sec = _prorated_encode_sec(per_set, set(query_ids))

    matches, index_build_sec, search_sec = _register_and_search(
        ref_emb, ref_ids, [ref_family_of[d] for d in ref_ids],
        query_emb, query_ids, [query_family_of[d] for d in query_ids],
        faiss_config, top_k)

    timing = {
        # encode_sec values come from the cache records, prorated by the documents actually used
        "ref_encode_sec": ref_encode_sec, "query_encode_sec": query_encode_sec,
        "search_sec": search_sec, "index_build_sec": index_build_sec,
        "total_sec": perf_counter() - started,
        # inference = encoding the query + searching. Reference encoding and index build are
        # registration and are reported separately.
        "inference_total_sec": query_encode_sec + search_sec,
        "build_sec": ref_encode_sec + index_build_sec,
        "embed_source": "cached" if set(sources) == {"cached"} else ("live" if set(sources) == {"live"} else "mixed"),
        "n_overflow_sets": n_overflow_total,   # documents needing more than one window, across all sets
        "model_name": model_name, "max_seq_length": int(max_seq_length),
    }
    return matches, timing
