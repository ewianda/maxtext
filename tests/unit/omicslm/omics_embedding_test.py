from __future__ import annotations

import jax.numpy as jnp
import pytest

from maxtext.layers.omics import inject_omics_embeddings


def test_inject_omics_embeddings_replaces_multiple_placeholders_in_order():
  text_embeddings = jnp.arange(2 * 5 * 3, dtype=jnp.float32).reshape(2, 5, 3)
  input_ids = jnp.array([[1, 99, 2, 99, 3], [99, 4, 5, 6, 99]], dtype=jnp.int32)
  omics_embeddings = jnp.array(
      [
          [[100.0, 101.0, 102.0], [110.0, 111.0, 112.0]],
          [[200.0, 201.0, 202.0], [210.0, 211.0, 212.0]],
      ],
      dtype=jnp.float32,
  )

  result = inject_omics_embeddings(text_embeddings, input_ids, omics_embeddings, omics_token_id=99)

  assert jnp.allclose(result[0, 1], omics_embeddings[0, 0])
  assert jnp.allclose(result[0, 3], omics_embeddings[0, 1])
  assert jnp.allclose(result[1, 0], omics_embeddings[1, 0])
  assert jnp.allclose(result[1, 4], omics_embeddings[1, 1])
  assert jnp.allclose(result[0, 0], text_embeddings[0, 0])


def test_inject_omics_embeddings_supports_padded_omics_batches():
  text_embeddings = jnp.zeros((1, 4, 2), dtype=jnp.float32)
  input_ids = jnp.array([[5, 42, 6, 42]], dtype=jnp.int32)
  omics_embeddings = jnp.array([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]], dtype=jnp.float32)
  omics_mask = jnp.array([[True, True, False]])

  result = inject_omics_embeddings(text_embeddings, input_ids, omics_embeddings, omics_token_id=42, omics_mask=omics_mask)

  assert jnp.allclose(result[0, 1], jnp.array([1.0, 2.0]))
  assert jnp.allclose(result[0, 3], jnp.array([3.0, 4.0]))


def test_inject_omics_embeddings_requires_matching_placeholder_counts():
  text_embeddings = jnp.zeros((1, 3, 2), dtype=jnp.float32)
  input_ids = jnp.array([[7, 42, 8]], dtype=jnp.int32)
  omics_embeddings = jnp.zeros((1, 2, 2), dtype=jnp.float32)

  with pytest.raises(ValueError, match="exactly one projected omics vector"):
    inject_omics_embeddings(text_embeddings, input_ids, omics_embeddings, omics_token_id=42)
