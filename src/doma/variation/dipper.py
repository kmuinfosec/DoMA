"""DIPPER (Krishna et al., 2023) paraphrase generation — T5-XXL, adversarial tier.

Turns an original docs_df into a paraphrased variant docs_df.
    variant_type=dipper, variant_level=lex{L}_order{O}
    doc_id = {source_doc_id}__dipper_{level}_s{seed}

torch / transformers are imported lazily so the schema and metric paths run without a GPU stack;
4-bit and 8-bit loading need bitsandbytes (``uv sync --group dipper``).
``paraphrase_fn(list[str]) -> list[str]`` is injected, so the frame-building logic can be exercised
with a stub.

Generation samples (``do_sample=True``, ``top_p=0.75``), so a rewrite is reproducible only to the
extent the sampler is pinned: ``paraphrase`` seeds per document, ``paraphrase_batch`` once per
batch, so batched output depends on batch composition.

Note: DIPPER's control codes are the inverse of diversity (code = 100 - diversity), i.e. lex=60
is sent to the model as lexical=40.
Reference implementation: https://github.com/martiansideofthemoon/ai-detection-paraphrases
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import nltk
import pandas as pd
from tqdm.auto import tqdm

from doma.io import save_prepared_set
from doma.schemas import normalize_docs_df

# The diversity grid DIPPER was trained on (steps of 20).
_VALID_DIVERSITY = (0, 20, 40, 60, 80, 100)


# Make sure the nltk sentence tokenizer is available (nltk >= 3.9 ships punkt_tab).
def _ensure_punkt() -> None:
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                nltk.data.find(f"tokenizers/{resource}")
                return
            except Exception:
                continue


# DIPPER control string; the code is the inverse of the requested diversity.
def build_control(lex_diversity: int, order_diversity: int) -> str:
    return f"lexical = {100 - lex_diversity}, order = {100 - order_diversity}"


# Token count of a text, special tokens included.
def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True).input_ids)


# Fit control + (sentence-trimmed prefix) + window into the token budget.
# The oldest prefix sentences are dropped whole; sentences are never cut mid-way. If dropping the
# entire prefix still overflows, the text is returned as is.
def fit_input(tokenizer, control: str, prefix_sents: list[str], window_block: str, budget: int) -> tuple[str, int]:
    start = 0
    while True:
        prefix = " ".join(prefix_sents[start:]).strip()
        pieces = [control] + ([prefix] if prefix else []) + [window_block]
        text = " ".join(pieces)
        total = count_tokens(tokenizer, text)
        if total <= budget or start >= len(prefix_sents):
            return text, total
        start += 1


# DIPPER strength and generation settings. lex/order are 0-100 in steps of 20; the protocol
# standard is (60, 60).
@dataclass
class DipperConfig:
    lex_diversity: int = 60
    order_diversity: int = 60
    sent_interval: int = 3
    max_new_tokens: int = 512
    top_p: float = 0.75
    do_sample: bool = True
    max_input_tokens: int = 512   # declared input length of t5-v1_1-xxl; longer prefixes are trimmed
    seed: int = 42                # torch.manual_seed for reproducible sampling

    # Reject diversity values outside the trained grid.
    def __post_init__(self) -> None:
        for name, value in (("lex_diversity", self.lex_diversity), ("order_diversity", self.order_diversity)):
            if value not in _VALID_DIVERSITY:
                raise ValueError(f"{name} must be one of {_VALID_DIVERSITY}, got {value}")


# Wrapper around the DIPPER paraphraser (T5-XXL, 11B). torch/transformers load on first use.
class DipperParaphraser:
    # Load model and tokenizer (quant: 4bit nf4 by default | 8bit | none = fp16).
    def __init__(
        self,
        model_name: str = "kalpeshk2011/dipper-paraphraser-xxl",
        tokenizer_name: str = "google/t5-v1_1-xxl",
        quant: str = "4bit",
        verbose: bool = True,
    ) -> None:
        import torch
        from transformers import BitsAndBytesConfig, T5ForConditionalGeneration, T5Tokenizer

        started = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        # Pin quantized loading to GPU 0; the accelerate planner otherwise spills onto the CPU.
        gpu0 = {"": 0}
        if quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            model_kwargs = {"quantization_config": quant_config, "device_map": gpu0}
        elif quant == "8bit":
            model_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": gpu0}
        elif quant == "none":
            model_kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        else:
            raise ValueError(f"quant must be '4bit' | '8bit' | 'none', got {quant!r}")

        self.model = T5ForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        _ensure_punkt()
        if verbose:
            print(f"[dipper] {model_name} loaded in {time.time() - started:.1f}s "
                  f"(quant={quant}, device={self.model.device})")

    # Slide a cfg.sent_interval-sentence window (default 3) across the document, accumulating the
    # rewritten text as prefix.
    def paraphrase(self, input_text: str, cfg: DipperConfig, prefix: str = "") -> str:
        import torch

        with torch.inference_mode():
            torch.manual_seed(cfg.seed)   # every document starts from the same RNG state
            control = build_control(cfg.lex_diversity, cfg.order_diversity)
            sentences = nltk.sent_tokenize(" ".join(input_text.split()))
            prefix_sents = nltk.sent_tokenize(prefix) if prefix.strip() else []
            output_parts: list[str] = []
            for i in range(0, len(sentences), cfg.sent_interval):
                window = " ".join(sentences[i: i + cfg.sent_interval])
                window_block = f"<sent> {window} </sent>"
                input_str, _ = fit_input(self.tokenizer, control, prefix_sents, window_block, cfg.max_input_tokens)
                encoded = self.tokenizer([input_str], return_tensors="pt", truncation=True,
                                         max_length=cfg.max_input_tokens)
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                generated = self.model.generate(**encoded, do_sample=cfg.do_sample, top_p=cfg.top_p,
                                                top_k=None, max_new_tokens=cfg.max_new_tokens)
                text = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
                output_parts.append(text)
                prefix_sents.extend(nltk.sent_tokenize(text))
            return " ".join(output_parts).strip()

    # Paraphrase several documents in parallel; each keeps its own prefix and position, so no
    # context leaks between documents. Sampling is seeded once for the whole batch, so a given
    # document's rewrite depends on which documents share its batch — use paraphrase() when the
    # result has to be independent of batch composition.
    def paraphrase_batch(self, texts: list[str], cfg: DipperConfig) -> list[str]:
        import torch

        with torch.inference_mode():
            torch.manual_seed(cfg.seed)
            control = build_control(cfg.lex_diversity, cfg.order_diversity)
            docs = [{"sents": nltk.sent_tokenize(" ".join(t.split())), "pos": 0, "prefix_sents": [], "out": []}
                    for t in texts]
            while True:
                active = [d for d in docs if d["pos"] < len(d["sents"])]
                if not active:
                    break
                inputs = []
                for d in active:
                    window = " ".join(d["sents"][d["pos"]: d["pos"] + cfg.sent_interval])
                    window_block = f"<sent> {window} </sent>"
                    input_str, _ = fit_input(self.tokenizer, control, d["prefix_sents"],
                                             window_block, cfg.max_input_tokens)
                    inputs.append(input_str)
                encoded = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                                         max_length=cfg.max_input_tokens)
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                generated = self.model.generate(**encoded, do_sample=cfg.do_sample, top_p=cfg.top_p,
                                                top_k=None, max_new_tokens=cfg.max_new_tokens)
                outs = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                for d, out in zip(active, outs):
                    d["out"].append(out)
                    d["prefix_sents"].extend(nltk.sent_tokenize(out))
                    d["pos"] += cfg.sent_interval
            return [" ".join(d["out"]).strip() for d in docs]


# Original docs_df -> DIPPER variant docs_df.
# paraphrase_fn rewrites a list of texts in one call and batch_size decides how many documents
# go into each call, so the same function serves single-document and batched generation.
# ``sec`` is the call's wall time divided by the documents in it, so the field means the same
# thing at any batch size.
def build_dipper_variant_df(
    original_df: pd.DataFrame, paraphrase_fn, lex: int = 60, order: int = 60, seed: int = 42,
    batch_size: int = 1,
) -> pd.DataFrame:
    level = f"lex{lex}_order{order}"
    step = max(1, int(batch_size))
    records = list(original_df.itertuples(index=False))
    rows = []
    for start in tqdm(range(0, len(records), step), desc=f"dipper {level}", leave=False):
        chunk = records[start: start + step]
        texts = [str(r.text) for r in chunk]
        started = time.perf_counter()
        paraphrases = paraphrase_fn(texts)
        sec = round((time.perf_counter() - started) / len(chunk), 1)
        for row, source_text, para in zip(chunk, texts, paraphrases):
            rows.append({
                "doc_id": f"{row.doc_id}__dipper_{level}_s{seed}",
                "dataset": row.dataset,
                "family_id": row.family_id,
                "text": para,
                "variant_type": "dipper",
                "variant_level": level,
                "variant_seed": seed,
                "source_doc_id": row.doc_id,
                "meta_json": {"lex": lex, "order": order, "sec": sec,
                              "in_words": len(source_text.split()), "out_words": len(para.split())},
            })
    return normalize_docs_df(pd.DataFrame(rows))


# Store the variant set as {dataset}_original_dipper_lex{L}_order{O}_s{seed}.
def build_and_save_dipper_variant_set(
    original_df: pd.DataFrame, prepared_dir, dataset: str, paraphrase_fn,
    lex: int = 60, order: int = 60, seed: int = 42, batch_size: int = 1,
) -> str:
    df = build_dipper_variant_df(original_df, paraphrase_fn, lex, order, seed, batch_size)
    name = f"{dataset}_original_dipper_lex{lex}_order{order}_s{seed}"
    save_prepared_set(df, prepared_dir, name)
    return name
