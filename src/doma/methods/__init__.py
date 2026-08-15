"""Detection methods.

Every method exposes ``xxx_matches(reference_docs_df, query_docs_df, ...) -> (matches_df, timing)``
and returns the ``doma.matches.MATCH_COLUMNS`` schema, so they are all scored by the same
``evaluate_run`` and land in the same run directory layout.

- ``doma``   : one mean-pooled vector per document, cosine nearest neighbour
- ``bm25``   : lexical full-document retrieval baseline
- ``ssdeep`` : context-triggered piecewise fuzzy hashing baseline
"""
from doma.methods.doma import DEFAULT_MODEL, doma_matches, resolve_max_seq_length

# Method name -> builder(cfg) -> (method_fn, run_tag). run_tag becomes the leaf of the run directory.
METHOD_BUILDERS = {}


# doma builder — reads cfg.method_params (model / max_seq_length / dtype / batch_size, and
# device, which only resolves the window; encoding auto-selects its own device).
def _build_doma(cfg):
    p = cfg.method_params
    model = p.get("model", DEFAULT_MODEL)
    dtype = p.get("dtype", "bfloat16")
    batch = int(p.get("batch_size", 8))
    # Unset max_seq_length means "the model's own maximum". Resolve it here so the run directory
    # records the window that was actually used (the answer is cached, so no model load per run).
    max_seq = p.get("max_seq_length")
    max_seq = int(max_seq) if max_seq else resolve_max_seq_length(
        model, cfg.artifacts_dir, p.get("device"), dtype)
    # The embedding cache is keyed per prepared set, so runs over different variant sets share it.
    set_names = [cfg.resolved_original_set, *cfg.resolved_variant_sets]

    def method_fn(reference_df, query_df):
        return doma_matches(
            reference_df, query_df, set_names=set_names, prepared_dir=cfg.prepared_dir,
            model_name=model, max_seq_length=max_seq, dtype=dtype, batch_size=batch,
            artifacts_dir=cfg.artifacts_dir, faiss_config=cfg.faiss_config)

    return method_fn, f"{model.split('/')[-1]}__L{max_seq}"


# bm25 builder — full-document lexical retrieval, no parameters.
def _build_bm25(cfg):
    def method_fn(reference_df, query_df):
        from doma.methods.bm25 import bm25_matches

        return bm25_matches(reference_df, query_df)

    return method_fn, "bm25"


# ssdeep builder — fuzzy hash comparison over full documents, no parameters.
def _build_ssdeep(cfg):
    def method_fn(reference_df, query_df):
        from doma.methods.ssdeep import ssdeep_matches

        return ssdeep_matches(reference_df, query_df)

    return method_fn, "ssdeep"


METHOD_BUILDERS.update({"doma": _build_doma, "bm25": _build_bm25, "ssdeep": _build_ssdeep})

__all__ = ["doma_matches", "DEFAULT_MODEL", "METHOD_BUILDERS"]
