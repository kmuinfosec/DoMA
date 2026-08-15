# DoMA: Document Membership Attribution

DoMA is a document-embedding-based leak detection framework that registers each confidential
document as **exactly one vector** and screens outbound documents by cosine nearest-neighbour
lookup, so a leak that has been **rewritten** — paraphrased, translated, or revised — is still
traced back to the confidential document it derives from. The repository contains the proposed
method, two baselines, three evaluation datasets, and the code that rebuilds the result tables
and figures.

---

## Overview

Exact-match DLP (hashing, fingerprinting) fails the moment a leaked document is reworded, and
chunk-level retrieval turns one document into many competing units that then have to be voted back
together. DoMA avoids both by making the whole document the unit of registration *and* comparison:

- **DoMA (`doma`)** — one mean-pooled, L2-normalized embedding per document, indexed with FAISS
  HNSW. Documents longer than the encoder's context window are cut into consecutive windows,
  encoded, and averaged. The detection score is the cosine to the
  nearest registered document; the family that document belongs to is the predicted source.
- **BM25 (`bm25`)** — sparse lexical retrieval over full documents.
- **Fuzzy hash (`ssdeep`)** — context-triggered piecewise hashing over full documents.

All three methods return the same match table (the query document, its best-matching confidential
document, that document's family, and a `confidence` score), so detection and attribution are
measured identically for every method.

---

## Repository Structure

```
FINAL DLP PUBLIC/
├── configs/                     # one YAML per dataset: shared settings + the list of runs
│   ├── krapivin.yaml            #   paraphrase leak (DIPPER lex60 / order60)
│   ├── par3.yaml                #   translation leak (machine → human translation)
│   ├── casimir_raw.yaml         #   revision leak (v1 → latest revision)
│   └── casimir_dedup.yaml       #   same, with duplicate-forum removal (variant)
├── src/doma/
│   ├── cli.py                   # entry point; prepare / run / table / corpus / figure / dipper / quality
│   ├── matches.py               # the match table every method returns, and thresholding
│   ├── report.py                # scan runs/**/metrics.json and build the result tables
│   ├── figures.py               # the ROC panels, redrawn from per_query_eval.parquet
│   ├── schemas.py io.py ids.py instrument.py
│   ├── methods/
│   │   ├── doma.py              # proposed method: mean-pooled document embeddings
│   │   └── bm25.py ssdeep.py    # baselines
│   ├── pipeline/
│   │   ├── config.py            # RunConfig and the per-dataset defaults
│   │   └── run.py               # the single experiment entry point
│   ├── splits/core.py           # family-aware 50/50 confidential split
│   ├── index/faiss_hnsw.py      # FAISS HNSW over confidential document vectors
│   ├── retrieval/core.py        # top-k search
│   ├── metrics/core.py          # detection, attribution, PR/ROC-AUC, case tables
│   ├── datasets/                # krapivin.py par3.py casimir.py
│   └── variation/
│       ├── dipper.py            # DIPPER paraphrase generation
│       └── quality.py           # surface-level variation quality metrics
├── data/prepared/               # prepared parquet sets (built by `prepare`; not shipped)
└── artifacts/                   # runs, embedding cache, tables and figures (auto-created)
```

---

## Requirements

Python ≥ 3.12. Install dependencies with:

```bash
uv sync
```

`torch` is pulled from PyPI by default. On a CUDA machine, install the wheel matching your driver,
for example:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Rebuilding the datasets or regenerating paraphrases needs the optional groups:

```bash
uv sync --group data     # CASIMIR download and preparation
uv sync --group dipper   # DIPPER paraphrase generation (GPU)
```

---

## Dataset Format

No dataset is shipped. Obtain the sources, then build the prepared parquet sets. Every prepared
parquet under `data/prepared/` carries exactly these columns, one document per row:

| Column | Description |
|---|---|
| `doc_id` | Unique document id |
| `dataset` | Dataset name (`krapivin` / `par3` / `casimir`) |
| `family_id` | Groups every document derived from one source text — the unit of the 50/50 split |
| `text` | Document body |
| `version_index` | Order within a family (CASIMIR revisions 1…n; PAR3 machine translation 0, human translations 1…n; empty elsewhere) |
| `variant_type` | `original` \| `revision` \| `dipper` \| `human_translation` \| … |
| `variant_level` | Variation strength (e.g. `lex60_order60`) |
| `variant_seed` | Seed used to generate the variant |
| `source_doc_id` | The `doc_id` this document was derived from — pairs a variant with its original |
| `num_chars`, `num_words` | Derived from `text` |
| `meta_json` | Dataset-specific extras, JSON string |

Sources:

| Dataset | Source | Leak |
|---|---|---|
| Krapivin | `krapivin_test.json`: JSONL, one record per paper, `{name, title, abstract, fulltext, keywords}` | DIPPER paraphrase (lex 60 / order 60) |
| PAR3 | `par3.pkl` from the PAR3 release (Thai et al., 2022); volumes kept apart, one volume per family | Human translation of the same work |
| CASIMIR | HuggingFace `taln-ls2n/CASIMIR`, downloaded automatically | The paper's latest revision |

**CASIMIR** is used as downloaded: no duplicate-forum removal is applied, so a paper resubmitted
under several forums stays in the set as several families. `configs/casimir_dedup.yaml` runs the
same pipeline over the deduplicated sets if that preprocessing step is wanted.

Krapivin redistributions carrying a record layout other than the one above (e.g.
`{file_name, text, keyphrases}`) are rejected with an error rather than silently producing an
empty set.

---

## Usage

Everything is accessed through `python -m doma.cli`.

### Prepare

```bash
uv run python -m doma.cli prepare krapivin --src data/raw/krapivin_test.json
uv run python -m doma.cli prepare par3     --src data/raw/par3.pkl
uv run python -m doma.cli prepare casimir
```

Inspect the prepared set sizes before running anything:

```bash
uv run python -m doma.cli corpus
```

This also reports the share of documents that exceed the encoder's context window.

### Run

```bash
uv run python -m doma.cli run configs/krapivin.yaml
uv run python -m doma.cli run configs/par3.yaml
uv run python -m doma.cli run configs/casimir_raw.yaml
uv run python -m doma.cli run configs/casimir_dedup.yaml   # optional: deduplicated variant
```

Each config lists the runs to execute (three DoMA encoders plus the two baselines) and prints one
result line per run. Every run writes to `artifacts/runs/<run_ident>/<run_tag>/`:

```
config.json  metrics.json  matches.parquet  query_manifest.parquet
per_query_eval.parquet  errors_by_family.parquet  fp_pairs.parquet  splits_used.parquet
```

Document embeddings are cached per prepared set under `artifacts/cache/embeddings_doma/<model>/`,
so re-runs that only change the split or the variant combination reuse them and never re-encode.

### Table

```bash
uv run python -m doma.cli table --out artifacts/paper_table.csv
```

Rebuilds the result tables by scanning `artifacts/runs/**/metrics.json`, never from a cached
summary file. A table cell is one specific experiment, identified by dataset, variant combination
and `include_original_as_positive` — not by "whatever ran on this dataset". Runs on other axes are
excluded rather than allowed to compete for the cell, and if several distinct runs still match one
cell (two split seeds, or the same model at two window lengths), the table raises instead of
reporting the best of them. Pass `--inclorig` to report the verbatim-leak setting instead.

### Figure

```bash
uv run python -m doma.cli figure --out artifacts/figures/roc.pdf
```

Redraws the ROC panels from the per-query scores of those same runs and prints the AUROC per panel.

### Dipper (optional, GPU)

```bash
uv run python -m doma.cli dipper krapivin --quant none --batch 4
```

Rewrites `<dataset>_original` with DIPPER (T5-XXL, 11B) and writes
`<dataset>_original_dipper_lex<L>_order<O>_s<seed>` to `data/prepared/`.

### Quality (optional)

```bash
uv run python -m doma.cli quality krapivin_original krapivin_original_dipper_lex60_order60_s42
```

Reports how far the rewrite moved from the original in wording.

---

## Arguments

### `prepare`

| Argument | Default | Description |
|---|---|---|
| `dataset` | — | `krapivin`, `par3`, or `casimir` |
| `--src` | `None` | Raw source file (not needed for `casimir`) |
| `--prepared` | `data/prepared` | Output directory for the prepared parquet sets |

### `run`

| Argument | Default | Description |
|---|---|---|
| `config` | — | Path to a YAML config under `configs/` |

### `table`

| Argument | Default | Description |
|---|---|---|
| `--artifacts` | `artifacts` | Root scanned for `runs/**/metrics.json` |
| `--out` | `None` | Optional CSV output path |
| `--inclorig` | `false` | Report the runs where the confidential originals were also screened as positives (verbatim leak) |

### `corpus`

| Argument | Default | Description |
|---|---|---|
| `--prepared` | `data/prepared` | Prepared sets to summarize |
| `--artifacts` | `artifacts` | Root holding the cached window length |
| `--model` | `Qwen/Qwen3-Embedding-0.6B` | Encoder whose cached embedding metadata supplies the over-window share |
| `--dtype` | `bfloat16` | Encoder dtype |
| `--out` | `None` | Optional CSV output path |

### `figure`

| Argument | Default | Description |
|---|---|---|
| `--artifacts` | `artifacts` | Root scanned for the runs behind the panels |
| `--out` | `artifacts/figures/roc.pdf` | Output figure path |

### `dipper`

| Argument | Default | Description |
|---|---|---|
| `dataset` | — | Reads `<dataset>_original`, writes `<dataset>_original_dipper_lex<L>_order<O>_s<seed>` |
| `--prepared` | `data/prepared` | Prepared set directory |
| `--quant` | `none` | `none` = fp16; `4bit` / `8bit` trade throughput for memory |
| `--batch` | `1` | Documents rewritten per generate call; `>1` seeds sampling once per batch, so the output depends on batch composition |
| `--lex` | `60` | Lexical diversity, 0–100 in steps of 20 |
| `--order` | `60` | Order diversity, 0–100 in steps of 20 |
| `--seed` | `42` | Sampling seed |
| `--max-input` / `--max-new` | `512` / `512` | Input and generation token budget per segment |
| `--limit` | `0` | Rewrite only the first N documents (0 = all), for a timing probe |

### `quality`

| Argument | Default | Description |
|---|---|---|
| `original_set` | — | Prepared set holding the originals |
| `variant_set` | — | Prepared set holding the rewrites |
| `--prepared` | `data/prepared` | Prepared set directory |
| `--out` | `None` | Optional per-document parquet output path |

### Config keys (`configs/*.yaml`)

| Key | Default | Description |
|---|---|---|
| `dataset` | — | `krapivin`, `par3`, `casimir_dedup`, or `casimir_raw` |
| `original_set` | per-dataset default | Prepared set registered as the confidential pool |
| `variant_sets` | per-dataset default | Prepared sets supplying the rewrites |
| `include_original_as_positive` | `false` | `true` also screens the confidential originals themselves (verbatim exfiltration) |
| `split_seed` | `42` | Seed of the family-aware split |
| `split_ratio` | `0.5` | Fraction of families registered as confidential |
| `prepared_dir` / `splits_dir` / `artifacts_dir` | `data/prepared` / `data/splits` / `artifacts` | Input and output roots |
| `runs` | — | List of `{method, params}`; `method` is `doma`, `bm25`, or `ssdeep` |
| `runs[].params` (DoMA) | — | `model`, `dtype`, `batch_size`; `max_seq_length` left unset = the model's own window |

---

## Method Details

### DoMA (`doma`)

1. Tokenize the document **without truncation**.
2. If it fits in the embedding model's maximum token window, encode it directly.
3. Otherwise cut the body into consecutive windows (that maximum length minus the special tokens),
   encode each window, and take the **unweighted mean** of the (un-normalized) window embeddings.
4. L2-normalize, so an inner-product index returns cosine similarity.
5. Register the confidential vectors in a FAISS HNSW index. A query's `confidence` is the raw cosine
   to its nearest registered document, and that document's family is the predicted source.

The pooling window is the model's own token limit, read from the loaded model rather than
hard-coded; each window leaves room for the
special tokens the encoder re-adds. Long documents are windowed rather than truncated, and the
same code path serves long-context and short-context encoders alike. `confidence` is evaluated by
a threshold sweep, never by a fixed cut-off.

### BM25 (`bm25`)

Sparse lexical retrieval over full documents (`k1 = 1.5`, `b = 0.75`, top-200 candidates); the raw
BM25 score of the best-matching family is the confidence.

### Fuzzy hash (`ssdeep`)

Context-triggered piecewise hashing over full documents; the best pairwise hash similarity,
rescaled from ppdeep's 0–100 to 0–1, is the confidence.

---

## Experimental Protocol

Identical for every dataset and method:

- Families are split **50/50**. A family is one source text with everything derived from it, so no
  family is ever split across the two sides.
- The confidential half is **registered** and becomes the reference pool.
- **Positive queries** = rewritten versions of the confidential documents.
- **Benign queries** = the benign half, originals and rewrites alike.
- The confidential originals stay in the reference pool and are not screened as queries
  (`include_original_as_positive: false`). Setting it to `true` adds verbatim exfiltration.

| Dataset | Confidential side | Leak |
|---|---|---|
| Krapivin | Scientific papers | DIPPER paraphrase (lex 60 / order 60) |
| PAR3 | Machine translation of a literary work | Human translation of the same work |
| CASIMIR | v1 of a paper | Its latest revision |

---

## Model

Three embedding models are configured for DoMA, each run at its own native window:

| Model | Window | dtype | `batch_size` |
|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 32768 | `bfloat16` | 2 |
| `ibm-granite/granite-embedding-97m-multilingual-r2` | 32768 | `bfloat16` | 4 |
| `BAAI/bge-base-en-v1.5` | 512 | `bfloat16` | 32 |

Index and retrieval defaults:

| Hyperparameter | Value |
|---|---|
| Index | FAISS `IndexHNSWFlat`, `METRIC_INNER_PRODUCT` |
| `M` (neighbours per node) | `32` |
| `ef_construction` (build beam) | `200` |
| `ef_search` (query beam) | `128` |
| `top_k` (neighbours retrieved) | `1` |

`top_k` is `1` because a document is one vector: the nearest neighbour *is* the prediction, and
there is nothing to aggregate over runner-ups. Raising it only materializes neighbours for
inspection — FAISS returns them in descending score order, so the match and confidence are
unchanged.

`M`, `ef_construction` and `ef_search` control how closely the approximate search tracks exact
cosine.

---

## Evaluation Metrics

- **Best-F1 under an oracle threshold** — the confidence threshold is swept exhaustively and the
  best F1 is recorded (`detection_best.best_f1`, alongside `best_threshold`, `best_precision`,
  `best_recall`). This isolates how separable the scores are from how well an operating point
  happens to be chosen.
- **Source-attribution accuracy** — of **all** positive queries, the fraction that was both flagged
  and traced back to the correct confidential family
  (`attribution.family_acc_on_all_positive`). The same figure over *detected* positives only is
  written as `attribution.family_acc_on_detected_positive`.
- **`pr_auc` / `roc_auc`** — threshold-free separability, and the source of the ROC panels.

Both metrics are read at the oracle best-F1 threshold, so attribution inherits that
operating point, and both come from the same `metrics.json`.

---

## Variation Quality

Surface-level only — how far the rewrite moved from the original in wording. No semantic or
model-based score is computed here.

| Metric | Direction |
|---|---|
| `ngram_overlap_1/2/3`, `bleu`, `jaccard_tokens` | lower = more heavily rewritten |
| `norm_edit_distance` | higher = more heavily rewritten |

`bleu` is standard BLEU-4 (Papineni et al., 2002) of the variant against its original: clipped
n-gram precisions for n = 1..4, uniform weights, brevity penalty, and NLTK smoothing method 1 for
orders with no match at all. `ngram_overlap_n` is the share of the *variant's* distinct n-grams
that also occur in the original.

---

## Latency Accounting

Building the index is **registration**, not inference, so its cost is measured separately and never
added to the recorded inference latency:

```
inference latency  = query encoding + index search
registration cost  = reference encoding/hashing + index build   (build_sec)
```

Cached encodings report the originally measured time, prorated by the documents a run actually
used, so throughput stays reproducible across re-runs.
