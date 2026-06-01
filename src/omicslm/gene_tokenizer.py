"""Tokenizer augmentation utilities for gene-aware OmicsLM vocabularies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GeneTokenizerAugmentation:
  added_tokens: tuple[str, ...]
  new_embeddings: np.ndarray


def _get_vocab(tokenizer) -> dict[str, int]:
  if hasattr(tokenizer, "get_vocab"):
    return dict(tokenizer.get_vocab())
  raise TypeError("Tokenizer must provide a get_vocab() method.")


def _encode_gene_symbol(tokenizer, gene_symbol: str) -> list[int]:
  if hasattr(tokenizer, "encode"):
    token_ids = tokenizer.encode(gene_symbol, add_special_tokens=False)
    if token_ids:
      return list(token_ids)
  raise ValueError(f"Unable to tokenize gene symbol {gene_symbol!r} with the provided tokenizer.")


def compute_gene_token_initializers(
    tokenizer,
    embedding_table: np.ndarray,
    gene_symbols: Sequence[str],
) -> GeneTokenizerAugmentation:
  """Computes mean-initialized embeddings for gene symbols."""
  vocab = _get_vocab(tokenizer)
  embedding_table = np.asarray(embedding_table)

  added_tokens: list[str] = []
  new_embeddings: list[np.ndarray] = []
  for gene_symbol in gene_symbols:
    if gene_symbol in vocab:
      continue
    token_ids = _encode_gene_symbol(tokenizer, gene_symbol)
    new_embeddings.append(embedding_table[np.asarray(token_ids)].mean(axis=0))
    added_tokens.append(gene_symbol)
  if not new_embeddings:
    feature_dim = embedding_table.shape[-1]
    return GeneTokenizerAugmentation(added_tokens=tuple(), new_embeddings=np.zeros((0, feature_dim), dtype=embedding_table.dtype))
  return GeneTokenizerAugmentation(
      added_tokens=tuple(added_tokens),
      new_embeddings=np.stack(new_embeddings).astype(embedding_table.dtype, copy=False),
  )


def augment_tokenizer_with_gene_tokens(
    tokenizer,
    embedding_table: np.ndarray,
    gene_symbols: Iterable[str],
) -> GeneTokenizerAugmentation:
  """Adds gene symbols to a tokenizer and returns their initialized embeddings."""
  augmentation = compute_gene_token_initializers(tokenizer, embedding_table, tuple(gene_symbols))
  if augmentation.added_tokens and hasattr(tokenizer, "add_tokens"):
    num_added = tokenizer.add_tokens(list(augmentation.added_tokens))
    if num_added != len(augmentation.added_tokens):
      raise ValueError(
          f"Tokenizer reported adding {num_added} tokens, expected {len(augmentation.added_tokens)}."
      )
  return augmentation


def extend_embedding_table(embedding_table: np.ndarray, new_embeddings: np.ndarray) -> np.ndarray:
  """Appends newly initialized gene embeddings to an embedding table."""
  embedding_table = np.asarray(embedding_table)
  new_embeddings = np.asarray(new_embeddings, dtype=embedding_table.dtype)
  if new_embeddings.size == 0:
    return embedding_table
  return np.concatenate([embedding_table, new_embeddings], axis=0)
