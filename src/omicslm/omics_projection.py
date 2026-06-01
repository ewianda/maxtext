"""Linear projection layer for OmicsLM."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


def scaled_xavier_uniform(gain: float):
  base_initializer = nn.initializers.xavier_uniform()

  def init(key, shape, dtype=jnp.float32):
    return gain * base_initializer(key, shape, dtype)

  return init


class OmicsProjection(nn.Module):
  """Projects full omics vectors into the LLM embedding space."""

  input_dim: int
  hidden_size: int
  gain: float = 0.01
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, omics_vectors: jnp.ndarray) -> jnp.ndarray:
    kernel = self.param(
        "kernel",
        scaled_xavier_uniform(self.gain),
        (self.input_dim, self.hidden_size),
        self.param_dtype,
    )
    bias = self.param("bias", nn.initializers.zeros_init(), (self.hidden_size,), self.param_dtype)
    omics_vectors = jnp.asarray(omics_vectors, self.dtype)
    return jnp.matmul(omics_vectors, kernel.astype(self.dtype)) + bias.astype(self.dtype)
