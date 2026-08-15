"""RunConfig — everything that identifies one experiment, plus the per-dataset defaults.

Experimental protocol (identical for every dataset and method):
  families are split 50/50; one half is registered as confidential, the other stays benign.
  positive queries  = variants of the confidential documents (paraphrase / translation / revision)
  benign queries    = the benign half, originals and variants alike
Because the split is by family, no family ever appears on both sides.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from doma.ids import sanitize_piece
from doma.index.faiss_hnsw import FAISSHNSWConfig

# Implemented methods: the proposed one plus the two baselines.
ALL_METHODS: tuple[str, ...] = ("doma", "bm25", "ssdeep")

# Default original_set / variant_sets per dataset (used when the config leaves them unset).
DATASET_DEFAULTS: dict[str, dict] = {
    # Paraphrase: DIPPER (lex 60 / order 60) rewrites of every paper; the confidential half's
    # rewrites are the leaks, the benign half's are benign queries.
    "krapivin": {
        "original_set": "krapivin_original",
        "variant_sets": ["krapivin_original_dipper_lex60_order60_s42"],
    },
    # Translation: the confidential DB is the machine translation, the leak is a human translation
    # of the same work. Volumes are kept separate (``_split``), one book volume per family.
    "par3": {
        "original_set": "par3_gt_split",
        "variant_sets": ["par3_human_t1_split"],
    },
    # Revision: v1 of a paper is confidential, its latest revision is the leak. Uses the set
    # as downloaded (no duplicate-forum removal).
    "casimir_raw": {
        "original_set": "casimir_raw_v1_only",
        "variant_sets": ["casimir_raw_latest_only"],
    },
    # Optional variant: papers resubmitted under several forums are dropped rather than merged
    # (see datasets/casimir.py).
    "casimir_dedup": {
        "original_set": "casimir_dedup_v1_only",
        "variant_sets": ["casimir_dedup_latest_only"],
    },
}


# The full specification of one run (frozen, so nothing mutates it mid-run).
@dataclass(frozen=True)
class RunConfig:
    dataset: str

    prepared_dir: str | Path = "data/prepared"
    splits_dir: str | Path = "data/splits"
    artifacts_dir: str | Path = "artifacts"

    original_set: str | None = None
    variant_sets: tuple[str, ...] | None = None   # tuple, so the frozen config stays immutable

    split_seed: int = 42
    split_ratio: float = 0.5

    faiss_config: FAISSHNSWConfig = field(default_factory=FAISSHNSWConfig)

    method: str = "doma"
    method_params: dict = field(default_factory=dict)   # e.g. doma {model, max_seq_length, dtype, batch_size, device}
    # False (default): only variants are leaks. True also screens the confidential
    # originals themselves, i.e. verbatim exfiltration.
    include_original_as_positive: bool = False

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_DEFAULTS:
            raise ValueError(f"unknown dataset {self.dataset!r}. choose from: {list(DATASET_DEFAULTS)}")
        if self.method not in ALL_METHODS:
            raise ValueError(f"unknown method {self.method!r}. choose from: {list(ALL_METHODS)}")

    # Resolved original_set (falls back to the dataset default).
    @property
    def resolved_original_set(self) -> str:
        return self.original_set or DATASET_DEFAULTS[self.dataset]["original_set"]

    # Resolved variant_sets (falls back to the dataset default).
    @property
    def resolved_variant_sets(self) -> list[str]:
        if self.variant_sets is not None:
            return list(self.variant_sets)
        return list(DATASET_DEFAULTS[self.dataset]["variant_sets"])

    # Fingerprint of a non-default variant combination, so an override cannot overwrite the
    # default run directory. The default combination gets no tag.
    def _variant_slug(self) -> str:
        resolved = tuple(self.resolved_variant_sets)
        if resolved == tuple(DATASET_DEFAULTS[self.dataset]["variant_sets"]):
            return ""
        digest = hashlib.sha1("|".join(resolved).encode("utf-8")).hexdigest()[:8]
        return f"__var{digest}"

    # Identity of the experiment = first level of runs/<run_ident>/.
    # Keyed by original_set rather than dataset: two original sets of one dataset have different
    # family universes, so they must never share a run directory.
    def run_ident(self, method: str | None = None) -> str:
        ident = f"{method or self.method}__{self.resolved_original_set}__s{self.split_seed}"
        ident += self._variant_slug()
        if self.include_original_as_positive:
            ident += "__inclorig"
        return sanitize_piece(ident)

    # dict for json/yaml dumps (resolved values and slugs included).
    def as_serializable(self) -> dict:
        d = asdict(self)
        d["prepared_dir"] = str(self.prepared_dir)
        d["splits_dir"] = str(self.splits_dir)
        d["artifacts_dir"] = str(self.artifacts_dir)
        if d.get("variant_sets") is not None:
            d["variant_sets"] = list(d["variant_sets"])
        d["_resolved"] = {
            "original_set": self.resolved_original_set,
            "variant_sets": self.resolved_variant_sets,
            "run_ident": self.run_ident(),
        }
        return d
