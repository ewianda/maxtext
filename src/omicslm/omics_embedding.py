"""Placeholder injection for OmicsLM omics embeddings."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

OMICS_PLACEHOLDER = "<omics>"


def build_omics_placeholder_mask(input_ids: jnp.ndarray, omics_token_id: int) -> jnp.ndarray:
  """Returns a mask for Omics placeholder token positions."""
  return jnp.asarray(input_ids == omics_token_id)


def _validate_placeholder_counts(placeholder_mask: jnp.ndarray, omics_mask: jnp.ndarray) -> None:
  placeholder_counts = np.asarray(placeholder_mask.sum(axis=1))
  omics_counts = np.asarray(omics_mask.sum(axis=1))
  if not np.array_equal(placeholder_counts, omics_counts):
    raise ValueError(
        "Each sequence must provide exactly one projected omics vector for every <omics> placeholder. "
        f"Got placeholders={placeholder_counts.tolist()} and omics={omics_counts.tolist()}."
    )


def inject_omics_embeddings(
    text_embeddings: jnp.ndarray,
    input_ids: jnp.ndarray,
    omics_embeddings: jnp.ndarray,
    omics_token_id: int,
    omics_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Replaces <omics> token embeddings with projected omics embeddings."""
  text_embeddings = jnp.asarray(text_embeddings)
  input_ids = jnp.asarray(input_ids)
  omics_embeddings = jnp.asarray(omics_embeddings)

  if text_embeddings.ndim != 3:
    raise ValueError(f"text_embeddings must have shape [batch, seq, hidden], got {text_embeddings.shape}.")
  if omics_embeddings.ndim != 3:
    raise ValueError(f"omics_embeddings must have shape [batch, num_omics, hidden], got {omics_embeddings.shape}.")
  if text_embeddings.shape[0] != omics_embeddings.shape[0]:
    raise ValueError("text_embeddings and omics_embeddings must have the same batch dimension.")
  if text_embeddings.shape[-1] != omics_embeddings.shape[-1]:
    raise ValueError("Embedding hidden sizes for text and omics inputs must match.")

  placeholder_mask = build_omics_placeholder_mask(input_ids, omics_token_id)
  if omics_mask is None:
    omics_mask = jnp.ones(omics_embeddings.shape[:2], dtype=bool)
  else:
    omics_mask = jnp.asarray(omics_mask, dtype=bool)

  _validate_placeholder_counts(placeholder_mask, omics_mask)

  def inject_single(text_row, placeholder_row, omics_row, omics_row_mask):
    order = jnp.argsort(-omics_row_mask.astype(jnp.int32))
    sorted_omics = omics_row[order]
    sorted_mask = omics_row_mask[order]
    target_positions = jnp.nonzero(placeholder_row, size=omics_row.shape[0], fill_value=0)[0]

    def body_fn(i, acc):
      return jax.lax.cond(
          sorted_mask[i],
          lambda current: current.at[target_positions[i]].set(sorted_omics[i]),
          lambda current: current,
          acc,
      )

    return jax.lax.fori_loop(0, omics_row.shape[0], body_fn, text_row)

  return jax.vmap(inject_single)(text_embeddings, placeholder_mask, omics_embeddings, omics_mask)
