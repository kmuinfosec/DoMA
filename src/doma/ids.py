"""doc_id / family_id construction rules.

A query is uniquely identified by its doc_id, so no separate query id exists.
"""
from __future__ import annotations

import re


# Normalize an id fragment: lowercase, spaces to _, drop anything outside [a-z0-9_@.\-:].
# ':' is kept because it is the family/doc id separator.
def sanitize_piece(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-zA-Z0-9_@.\-:]+", "", value)
    return value


# Build the id of a family (one source text plus everything derived from it): '<dataset>:<family_key>'.
def build_family_id(dataset: str, family_key: str) -> str:
    return f"{sanitize_piece(dataset)}:{sanitize_piece(family_key)}"


# Build the id of a single document inside a family: '<family_id>:<doc_key>'.
def build_doc_id(family_id: str, doc_key: str) -> str:
    return f"{sanitize_piece(family_id)}:{sanitize_piece(doc_key)}"


# doc_key of a revision document: 'v1', 'v2', ... (used by CASIMIR).
def make_revision_doc_key(version_index: int) -> str:
    return f"v{int(version_index)}"


# Hugging Face model name -> path-safe slug, used as an embedding cache directory name.
def model_slug(name: str) -> str:
    return name.replace("/", "__").replace(":", "_").replace("@", "_")
