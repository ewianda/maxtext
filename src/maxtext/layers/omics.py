# Copyright 2024-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OmicsLM integration layers for MaxText.

Provides:
  OmicsProjection  – a linear layer that maps raw omics vectors into the LLM
                     embedding space with a small-gain initializer.
  inject_omics_embeddings – replaces <omics> placeholder token embeddings with
                            the projected omics vectors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np


def _scaled_xavier_uniform(gain: float):
  """Returns an NdInitializer that scales xavier_uniform by *gain*."""
  base = nn.initializers.xavier_uniform()

  def init(key, shape, dtype=jnp.float32, in_axis=0, out_axis=1):
    return gain * base(key, shape, dtype)

  return init


class OmicsProjection(nn.Module):
  """Projects raw omics vectors into the LLM embedding space.

  This is a single linear transformation (affine projection) with a
  small-gain initializer to keep omics contributions stable at the start
  of training while remaining fully checkpoint-compatible with Linen/Orbax.

  Attributes:
    input_dim:   Dimensionality of the raw omics input (default 20006 for
                 expression-only from GENCODE v47 intersect Geneformer V2).
    hidden_size: Dimensionality of the LLM embedding space.
    gain:        Scaling factor for the Xavier-uniform weight initializer.
    dtype:       Compute dtype (inherits from config dtype).
    weight_dtype: Storage dtype for the kernel.
  """

  input_dim: int
  hidden_size: int
  gain: float = 0.01
  dtype: jnp.dtype = jnp.bfloat16
  weight_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, omics_vectors: jnp.ndarray) -> jnp.ndarray:
    """Project omics vectors.

    Args:
      omics_vectors: float array of shape [batch, num_omics, input_dim].

    Returns:
      Projected array of shape [batch, num_omics, hidden_size].
    """
    return nn.Dense(
        features=self.hidden_size,
        use_bias=True,
        kernel_init=_scaled_xavier_uniform(self.gain),
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        name="omics_kernel",
    )(jnp.asarray(omics_vectors, self.dtype))


def inject_omics_embeddings(
    text_embeddings: jnp.ndarray,
    input_ids: jnp.ndarray,
    omics_embeddings: jnp.ndarray,
    omics_token_id: int,
    omics_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Replace <omics> placeholder positions with projected omics embeddings.

  For each sequence in the batch the function finds positions where
  ``input_ids == omics_token_id`` and writes the corresponding projected
  omics vector into ``text_embeddings``.  Sequences may have a different
  number of placeholders – the function uses ``jax.lax.dynamic_update_slice``
  style updates via ``jax.vmap`` so it is fully JIT-compatible.

  Args:
    text_embeddings:  [B, T, H] array produced by the token embedder.
    input_ids:        [B, T]   integer token IDs.
    omics_embeddings: [B, N, H] projected omics vectors (N ≤ T).
    omics_token_id:   integer token ID that marks placeholder positions.
    omics_mask:       optional [B, N] mask for padded omics batches.

  Returns:
    Updated text_embeddings [B, T, H] with placeholder positions replaced.
  """
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

  # Cast to the text embedding dtype so there is no dtype mismatch.
  omics_embeddings = omics_embeddings.astype(text_embeddings.dtype)

  placeholder_mask = input_ids == omics_token_id
  if omics_mask is None:
    omics_mask = jnp.ones(omics_embeddings.shape[:2], dtype=bool)
  else:
    omics_mask = jnp.asarray(omics_mask, dtype=bool)

  placeholder_counts = placeholder_mask.sum(axis=1)
  omics_counts = omics_mask.sum(axis=1)
  # This validation is best-effort for eager execution. During tracing we skip
  # the Python exception path so the injection logic remains JIT-compatible.
  try:
    placeholder_counts_np = np.asarray(placeholder_counts)
    omics_counts_np = np.asarray(omics_counts)
  except jax.errors.ConcretizationTypeError:
    placeholder_counts_np = None
    omics_counts_np = None

  if placeholder_counts_np is not None and not np.array_equal(placeholder_counts_np, omics_counts_np):
    raise ValueError(
        "Each sequence must provide exactly one projected omics vector for every <omics> placeholder. "
        f"Got placeholders={placeholder_counts_np.tolist()} and omics={omics_counts_np.tolist()}."
    )

  def _inject_row(text_row, placeholder_row, omics_row, omics_row_mask):
    order = jnp.argsort(-omics_row_mask.astype(jnp.int32))
    sorted_omics = omics_row[order]
    sorted_mask = omics_row_mask[order]
    positions = jnp.nonzero(placeholder_row, size=omics_row.shape[0], fill_value=0)[0]

    def _set_one(acc, args):
      pos, vec, mask = args
      acc = jax.lax.cond(
          mask,
          lambda current: jax.lax.dynamic_update_slice(current, vec[jnp.newaxis, :], (pos, 0)),
          lambda current: current,
          acc,
      )
      return acc, None

    text_row, _ = jax.lax.scan(_set_one, text_row, (positions, sorted_omics, sorted_mask))
    return text_row

  return jax.vmap(_inject_row)(text_embeddings, placeholder_mask, omics_embeddings, omics_mask)
