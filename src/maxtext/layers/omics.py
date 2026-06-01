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
  OmicsProjection  – a sharding-aware linear layer that maps raw omics vectors
                     (e.g. expression + FunOmics + Geneformer concatenation)
                     into the LLM embedding space.
  inject_omics_embeddings – replaces <omics> placeholder token embeddings with
                            the projected omics vectors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from maxtext.layers.linears import DenseGeneral


def _scaled_xavier_uniform(gain: float):
  """Returns an initializer that scales xavier_uniform by *gain*."""
  base = nn.initializers.xavier_uniform()

  def init(key, shape, dtype=jnp.float32):
    return gain * base(key, shape, dtype)

  return init


class OmicsProjection(nn.Module):
  """Projects raw omics vectors into the LLM embedding space.

  This is a single linear transformation (affine projection) with a
  small-gain initializer to keep omics contributions stable at the start
  of training.  The kernel is annotated with the 'embed' axis so that it
  participates in MaxText's standard FSDP sharding rules.

  Attributes:
    input_dim:   Dimensionality of the raw omics input (e.g. 20541).
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
    return DenseGeneral(
        in_features_shape=self.input_dim,
        out_features_shape=self.hidden_size,
        axis=-1,
        kernel_init=_scaled_xavier_uniform(self.gain),
        kernel_axes=("embed", "mlp"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        use_bias=True,
        name="omics_kernel",
    )(jnp.asarray(omics_vectors, self.dtype))


def inject_omics_embeddings(
    text_embeddings: jnp.ndarray,
    input_ids: jnp.ndarray,
    omics_embeddings: jnp.ndarray,
    omics_token_id: int,
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

  Returns:
    Updated text_embeddings [B, T, H] with placeholder positions replaced.
  """
  text_embeddings = jnp.asarray(text_embeddings)
  input_ids = jnp.asarray(input_ids)
  omics_embeddings = jnp.asarray(omics_embeddings)

  # Cast to the text embedding dtype so there is no dtype mismatch.
  omics_embeddings = omics_embeddings.astype(text_embeddings.dtype)

  n_omics = omics_embeddings.shape[1]
  placeholder_mask = (input_ids == omics_token_id)  # [B, T] bool

  def _inject_row(text_row, placeholder_row, omics_row):
    # Gather positions of placeholders (at most n_omics used).
    positions = jnp.nonzero(placeholder_row, size=n_omics, fill_value=0)[0]  # [N]

    def _set_one(acc, args):
      pos, vec = args
      acc = jax.lax.dynamic_update_slice(acc, vec[jnp.newaxis, :], (pos, 0))
      return acc, None

    text_row, _ = jax.lax.scan(_set_one, text_row, (positions, omics_row))
    return text_row

  return jax.vmap(_inject_row)(text_embeddings, placeholder_mask, omics_embeddings)
