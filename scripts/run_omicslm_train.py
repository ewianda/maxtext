#!/usr/bin/env python3
# Copyright 2024–2026 Google LLC
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

"""End-to-end OmicsLM instruction-tuning training script.

Runs a self-contained training loop using synthetic data so that the
pipeline can be validated on any GPU or TPU without external datasets.

Usage (see scripts/README.md for full details):

    # Minimal smoke run (CPU/GPU/TPU — no external data needed):
    python scripts/run_omicslm_train.py

    # Custom hyperparameters:
    python scripts/run_omicslm_train.py \
        --batch_size 4 \
        --seq_len 128 \
        --steps 200 \
        --learning_rate 1e-4 \
        --omics_dim 20541 \
        --hidden_size 256 \
        --output_dir /tmp/omicslm_out
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator

import numpy as np

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from omicslm.model import OmicsLM
from omicslm.config import OmicsLMConfig


# ---------------------------------------------------------------------------
# Tiny backbone (no external checkpoint needed)
# ---------------------------------------------------------------------------

class _TinyBackbone(nn.Module):
  """Minimal causal LM backbone for smoke/integration testing."""

  vocab_size: int
  hidden_size: int
  num_layers: int = 2

  @nn.compact
  def __call__(self, input_ids: jnp.ndarray, inputs_embeds: jnp.ndarray | None = None, **_):
    if inputs_embeds is None:
      x = nn.Embed(self.vocab_size, self.hidden_size)(input_ids)
    else:
      x = inputs_embeds
    for _ in range(self.num_layers):
      residual = x
      x = nn.LayerNorm()(x)
      x = nn.Dense(self.hidden_size)(x)
      x = nn.gelu(x)
      x = nn.Dense(self.hidden_size)(x)
      x = x + residual
    logits = nn.Dense(self.vocab_size, use_bias=False)(x)
    return {"logits": logits}

  # Expose the token embedder so OmicsLM can call it.
  def token_embedder(self, input_ids: jnp.ndarray) -> jnp.ndarray:
    return nn.Embed(self.vocab_size, self.hidden_size)(input_ids)


class _TinyBackboneWithEmbedder(nn.Module):
  """Wraps _TinyBackbone so OmicsLM can access shared_embedding."""

  vocab_size: int
  hidden_size: int
  num_layers: int = 2

  def setup(self):
    self._embed = nn.Embed(self.vocab_size, self.hidden_size)
    self._layers = [
        (nn.LayerNorm(), nn.Dense(self.hidden_size), nn.Dense(self.hidden_size))
        for _ in range(self.num_layers)
    ]
    self._head = nn.Dense(self.vocab_size, use_bias=False)

  def shared_embedding(self, input_ids: jnp.ndarray) -> jnp.ndarray:
    return self._embed(input_ids)

  def __call__(self, input_ids: jnp.ndarray, inputs_embeds: jnp.ndarray | None = None, **_):
    x = self._embed(input_ids) if inputs_embeds is None else inputs_embeds
    for ln, ff1, ff2 in self._layers:
      residual = x
      x = ln(x)
      x = ff2(nn.gelu(ff1(x))) + residual
    return {"logits": self._head(x)}


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def _synthetic_batch(
    *,
    batch_size: int,
    seq_len: int,
    omics_dim: int,
    vocab_size: int,
    omics_token_id: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
  """Returns a batch with exactly one <omics> placeholder per sequence."""
  # Random token IDs — reserve 0 for pad, omics_token_id for the placeholder.
  input_ids = rng.integers(1, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
  # Place the <omics> placeholder at position 1 (after a fake BOS).
  input_ids[:, 0] = 1  # BOS
  input_ids[:, 1] = omics_token_id
  labels = input_ids.copy()
  labels[:, :2] = -100  # ignore BOS and <omics> positions in the loss

  omics_inputs = rng.standard_normal((batch_size, 1, omics_dim)).astype(np.float32)
  return {
      "input_ids": input_ids,
      "labels": labels,
      "omics_inputs": omics_inputs,
  }


def _data_iterator(
    *,
    batch_size: int,
    seq_len: int,
    omics_dim: int,
    vocab_size: int,
    omics_token_id: int,
    seed: int = 0,
) -> Iterator[dict[str, np.ndarray]]:
  rng = np.random.default_rng(seed)
  while True:
    yield _synthetic_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        omics_dim=omics_dim,
        vocab_size=vocab_size,
        omics_token_id=omics_token_id,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
  print(f"JAX devices: {jax.devices()}")
  print(f"Training OmicsLM for {args.steps} steps  |  batch={args.batch_size}  seq={args.seq_len}")

  cfg = OmicsLMConfig(
      omics_dim=args.omics_dim,
      hidden_size=args.hidden_size,
      omics_token_id=args.omics_token_id,
      learning_rate=args.learning_rate,
      batch_size=args.batch_size,
  )

  backbone = _TinyBackboneWithEmbedder(
      vocab_size=args.vocab_size,
      hidden_size=args.hidden_size,
      num_layers=args.num_layers,
  )

  model = OmicsLM(
      backbone=backbone,
      omics_dim=cfg.expected_omics_dim,
      hidden_size=cfg.hidden_size,
      omics_token_id=cfg.omics_token_id,
      projection_init_gain=cfg.projection_init_gain,
  )

  # Initialise with a dummy batch so JAX traces the model.
  dummy_ids = jnp.zeros((1, args.seq_len), dtype=jnp.int32)
  dummy_omics = jnp.zeros((1, 1, cfg.expected_omics_dim), dtype=jnp.float32)
  params = model.init(jax.random.PRNGKey(0), dummy_ids, omics_inputs=dummy_omics)
  param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
  print(f"Model parameters: {param_count:,}")

  optimizer = optax.adam(args.learning_rate)
  opt_state = optimizer.init(params)

  @jax.jit
  def train_step(params, opt_state, batch):
    def loss_fn(p):
      out = model.apply(
          p,
          jnp.asarray(batch["input_ids"]),
          omics_inputs=jnp.asarray(batch["omics_inputs"]),
          labels=jnp.asarray(batch["labels"]),
      )
      return out.loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

  data_iter = _data_iterator(
      batch_size=args.batch_size,
      seq_len=args.seq_len,
      omics_dim=cfg.expected_omics_dim,
      vocab_size=args.vocab_size,
      omics_token_id=cfg.omics_token_id,
  )

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  t0 = time.time()
  for step in range(1, args.steps + 1):
    batch = next(data_iter)
    params, opt_state, loss = train_step(params, opt_state, batch)
    if step % args.log_every == 0 or step == 1:
      elapsed = time.time() - t0
      print(f"step {step:>5d}/{args.steps}  loss={float(loss):.4f}  elapsed={elapsed:.1f}s")

  print(f"\nTraining complete in {time.time() - t0:.1f}s")

  # Save final params as a .npz checkpoint.
  ckpt_path = output_dir / "omicslm_params.npz"
  flat_params = {
      "/".join(map(str, path)): np.asarray(leaf)
      for path, leaf in jax.tree_util.tree_leaves_with_path(params)
  }
  np.savez(str(ckpt_path), **flat_params)
  print(f"Checkpoint saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="OmicsLM end-to-end training smoke script.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--batch_size", type=int, default=2)
  parser.add_argument("--seq_len", type=int, default=64)
  parser.add_argument("--steps", type=int, default=50)
  parser.add_argument("--log_every", type=int, default=10)
  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--omics_dim", type=int, default=21287,
                      help="Full omics input dimension (must equal 1+expression+funomics+geneformer). "
                           "Default 21287 = 1 + 20006 (GENCODE v47 ∩ Geneformer V2) + 512 + 768.")
  parser.add_argument("--hidden_size", type=int, default=64,
                      help="LLM hidden dimension.  Use 4096 for Llama-2-7B.")
  parser.add_argument("--num_layers", type=int, default=2,
                      help="Number of backbone transformer layers (smoke default=2).")
  parser.add_argument("--vocab_size", type=int, default=1024)
  parser.add_argument("--omics_token_id", type=int, default=32,
                      help="Token ID used as the <omics> placeholder in input_ids.")
  parser.add_argument("--output_dir", type=str, default="/tmp/omicslm_out")
  return parser.parse_args()


if __name__ == "__main__":
  train(_parse_args())
