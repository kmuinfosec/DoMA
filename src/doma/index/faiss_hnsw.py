"""FAISS HNSW index over confidential document vectors — build / search.

Vectors are L2-normalized, so inner product equals cosine and the metric is fixed to inner product.
Building the index is registration, not inference: its cost is measured but excluded from the
recorded inference latency.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd


# HNSW hyperparameters (M = neighbours per node, efConstruction = build beam, efSearch = query beam).
@dataclass(frozen=True)
class FAISSHNSWConfig:
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 128


# A built index bundled with its document metadata and configuration.
class FAISSHNSWArtifact:
    def __init__(self, index, meta_df: pd.DataFrame, config: FAISSHNSWConfig) -> None:
        self.index = index
        self.meta_df = meta_df.reset_index(drop=True)
        self.config = config

    # Search top_k neighbours for query vectors (N x D) -> (scores, indices), both N x top_k.
    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(queries.astype(np.float32), top_k)


# Build an HNSW index from document vectors. meta_df must carry doc_id and family_id, one row
# per vector, and the vectors must already be L2-normalized (inner product = cosine).
def build_faiss_hnsw_index(
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    config: FAISSHNSWConfig | None = None,
) -> FAISSHNSWArtifact:
    config = config or FAISSHNSWConfig()
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D")
    if len(meta_df) != embeddings.shape[0]:
        raise ValueError("meta_df row count must match the number of embeddings")

    embeddings = embeddings.astype(np.float32)
    index = faiss.IndexHNSWFlat(embeddings.shape[1], config.M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = config.ef_construction
    index.hnsw.efSearch = config.ef_search
    index.add(embeddings)
    return FAISSHNSWArtifact(index=index, meta_df=meta_df, config=config)
