from __future__ import annotations

import jax
import jax.numpy as jnp

from omicslm.omics_projection import OmicsProjection


def test_omics_projection_shapes_and_zero_bias():
  module = OmicsProjection(input_dim=4, hidden_size=3, gain=0.01)
  inputs = jnp.ones((2, 4), dtype=jnp.float32)
  variables = module.init(jax.random.key(0), inputs)

  assert variables["params"]["kernel"].shape == (4, 3)
  assert variables["params"]["bias"].shape == (3,)
  assert jnp.allclose(variables["params"]["bias"], jnp.zeros((3,), dtype=jnp.float32))

  outputs = module.apply(variables, inputs)
  assert outputs.shape == (2, 3)
