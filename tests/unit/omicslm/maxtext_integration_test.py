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

"""Tests for OmicsLM integration into the MaxText Transformer.

Verifies that:
  1. Enabling `use_omics=True` in the config correctly wires up the
     OmicsProjection sub-module inside the MaxText Decoder.
  2. Model initialisation with omics inputs succeeds and produces extra
     `omics_projection` parameters.
  3. A forward pass with and without omics inputs completes without error.
  4. The omics projection changes the hidden states at placeholder positions
     (i.e. injection actually happens).
"""

import sys
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN, DECODING_ACTIVE_SEQUENCE_INDICATOR
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path


# A token ID large enough to never appear in randomly generated sequences but
# small enough to be inside any vocab we use for testing.
_OMICS_TOKEN_ID = 5
_OMICS_DIM = 8   # tiny for fast unit tests


def _init_omics_config(**kwargs):
  """Return a pyconfig for a tiny model with use_omics=True."""
  return pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      per_device_batch_size=1.0,
      run_name="omicslm_test",
      enable_checkpointing=False,
      base_num_decoder_layers=2,
      attention="dot_product",
      max_target_length=16,
      base_emb_dim=32,
      base_num_query_heads=2,
      base_num_kv_heads=2,
      max_prefill_predict_length=4,
      use_omics=True,
      omics_dim=_OMICS_DIM,
      omics_token_id=_OMICS_TOKEN_ID,
      omics_projection_gain=0.01,
      **kwargs,
  )


class TestOmicsIntegration(unittest.TestCase):
  """End-to-end MaxText integration tests for OmicsLM."""

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(42)
    self.cfg = _init_omics_config()
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.model = models.transformer_as_linen(
        config=self.cfg,
        mesh=self.mesh,
        quant=None,
        model_mode=MODEL_MODE_TRAIN,
    )

  def _make_batch(self, inject_omics: bool = True):
    """Create a minimal batch of tokens with one <omics> placeholder per row."""
    B = self.cfg.global_batch_size_to_train_on
    T = self.cfg.max_target_length
    N = 1  # one omics vector per sequence

    rng, key = jax.random.split(self.rng)
    ids = jax.random.randint(key, (B, T), 1, self.cfg.vocab_size)
    # Put an <omics> placeholder at position 2 in every sequence.
    ids = ids.at[:, 2].set(_OMICS_TOKEN_ID)

    positions = jnp.tile(jnp.arange(T, dtype=jnp.int32), (B, 1))
    segment_ids = jnp.ones((B, T), dtype=jnp.int32) * DECODING_ACTIVE_SEQUENCE_INDICATOR

    omics = None
    if inject_omics:
      rng, key = jax.random.split(rng)
      omics = jax.random.normal(key, (B, N, _OMICS_DIM), dtype=jnp.float32)

    return ids, positions, segment_ids, omics

  def test_model_init_creates_omics_projection_params(self):
    """OmicsProjection parameters appear under the expected path."""
    ids, positions, segment_ids, omics = self._make_batch()
    init_rng = {"params": self.rng, "aqt": self.rng, "dropout": self.rng}
    params = self.model.init(init_rng, ids, positions, segment_ids, omics_inputs=omics, enable_dropout=False)
    # Find the omics_projection kernel anywhere in the param tree.
    flat = jax.tree_util.tree_leaves_with_path(params)
    paths = ["/".join(str(k) for k in path) for path, _ in flat]
    self.assertTrue(
        any("omics_projection" in p for p in paths),
        msg=f"omics_projection not found in param tree. Got: {paths[:20]}",
    )

  def test_forward_pass_with_omics(self):
    """Forward pass with omics inputs does not raise and returns correct shape."""
    ids, positions, segment_ids, omics = self._make_batch(inject_omics=True)
    init_rng = {"params": self.rng, "aqt": self.rng, "dropout": self.rng}
    params = self.model.init(init_rng, ids, positions, segment_ids, omics_inputs=omics, enable_dropout=False)
    logits = self.model.apply(
        params,
        ids,
        positions,
        segment_ids,
        omics_inputs=omics,
        enable_dropout=False,
        model_mode=MODEL_MODE_TRAIN,
    )
    expected_shape = (self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length, self.cfg.vocab_size)
    self.assertEqual(logits.shape, expected_shape)

  def test_forward_pass_without_omics_inputs(self):
    """If omics_inputs=None the model still runs (placeholder positions get text embeddings)."""
    ids, positions, segment_ids, _ = self._make_batch(inject_omics=False)
    init_rng = {"params": self.rng, "aqt": self.rng, "dropout": self.rng}
    # Initialise with omics so the projection params exist.
    _, _, _, omics_dummy = self._make_batch(inject_omics=True)
    params = self.model.init(init_rng, ids, positions, segment_ids, omics_inputs=omics_dummy, enable_dropout=False)
    logits = self.model.apply(
        params,
        ids,
        positions,
        segment_ids,
        omics_inputs=None,
        enable_dropout=False,
        model_mode=MODEL_MODE_TRAIN,
    )
    expected_shape = (self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length, self.cfg.vocab_size)
    self.assertEqual(logits.shape, expected_shape)

  def test_omics_injection_changes_logits(self):
    """Injecting omics embeddings changes the logits at placeholder positions."""
    ids, positions, segment_ids, omics = self._make_batch(inject_omics=True)
    init_rng = {"params": self.rng, "aqt": self.rng, "dropout": self.rng}
    params = self.model.init(init_rng, ids, positions, segment_ids, omics_inputs=omics, enable_dropout=False)

    logits_with_omics = self.model.apply(
        params, ids, positions, segment_ids, omics_inputs=omics, enable_dropout=False, model_mode=MODEL_MODE_TRAIN
    )
    logits_without_omics = self.model.apply(
        params, ids, positions, segment_ids, omics_inputs=None, enable_dropout=False, model_mode=MODEL_MODE_TRAIN
    )
    # Unless the projection weights are exactly zero, the logits differ.
    self.assertFalse(
        jnp.allclose(logits_with_omics, logits_without_omics),
        msg="Logits should differ when omics are injected vs not.",
    )

  def test_no_omics_config_unaffected(self):
    """When use_omics=False the model behaves as before (no extra params)."""
    cfg_no_omics = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="no_omics_test",
        enable_checkpointing=False,
        base_num_decoder_layers=2,
        attention="dot_product",
        max_target_length=16,
        base_emb_dim=32,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        max_prefill_predict_length=4,
        use_omics=False,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg_no_omics)
    mesh = Mesh(devices_array, cfg_no_omics.mesh_axes)
    model = models.transformer_as_linen(config=cfg_no_omics, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
    ids, positions, segment_ids, _ = self._make_batch(inject_omics=False)
    init_rng = {"params": self.rng, "aqt": self.rng, "dropout": self.rng}
    params = model.init(init_rng, ids, positions, segment_ids, enable_dropout=False)
    flat = jax.tree_util.tree_leaves_with_path(params)
    paths = ["/".join(str(k) for k in path) for path, _ in flat]
    self.assertFalse(
        any("omics_projection" in p for p in paths),
        msg="omics_projection should not exist when use_omics=False",
    )


if __name__ == "__main__":
  unittest.main()
