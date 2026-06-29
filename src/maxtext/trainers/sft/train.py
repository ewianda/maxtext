"""Tunix-free SFT trainer entrypoint for MaxText.

This script provides a minimal supervised fine-tuning (SFT) loop using MaxText
core model/optimizer/checkpoint infra and a local JSONL pipeline.

It intentionally avoids Tunix dependencies.
"""

from __future__ import annotations

from typing import Any, Iterator

from absl import app
import jax
import jax.numpy as jnp
import numpy as np
import optax
import transformers

from maxtext.configs import pyconfig
from maxtext.data.sft_dataset import batch_examples, iter_sft_examples
from maxtext.optimizers import optimizers
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils


def _build_tokenizer(config: Any):
  tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer


def _cross_entropy_loss(logits: jax.Array, targets: jax.Array, weights: jax.Array) -> jax.Array:
  """Token-level weighted CE with stable normalization."""
  one_hot = jax.nn.one_hot(targets, logits.shape[-1], dtype=logits.dtype)
  per_token = optax.softmax_cross_entropy(logits, one_hot)
  weighted = per_token * weights
  denom = jnp.maximum(weights.sum(), 1.0)
  return weighted.sum() / denom


def _make_data_iterator(config: Any, tokenizer: Any) -> Iterator[dict[str, np.ndarray]]:
  examples = iter_sft_examples(
      path=config.sft_data_path,
      tokenizer=tokenizer,
      max_length=config.max_target_length,
      messages_field=getattr(config, "sft_messages_field", "messages"),
      prompt_field=getattr(config, "sft_prompt_field", "prompt"),
      response_field=getattr(config, "sft_response_field", "response"),
      text_field=getattr(config, "sft_text_field", "text"),
      use_chat_template=getattr(config, "use_chat_template", True),
      train_on_prompt=getattr(config, "sft_train_on_prompt", False),
  )
  return batch_examples(examples, batch_size=config.per_device_batch_size)


def main(argv: list[str]) -> None:
  del argv
  config = pyconfig.initialize(argv=[])

  if not getattr(config, "sft_data_path", None):
    raise ValueError("sft_data_path must be set for SFT training.")

  tokenizer = _build_tokenizer(config)

  # Model creation/restoration follows existing MaxText utility conventions.
  model, params, state, mesh, _ = model_creation_utils.initialize_model(config)

  # Optimizer via MaxText optimizer stack.
  tx = optimizers.get_optimizer(config)
  opt_state = tx.init(params)

  @jax.jit
  def train_step(params, state, opt_state, batch):
    def loss_fn(p):
      logits, new_state = model.apply(
          {"params": p, **state},
          decoder_input_tokens=batch["decoder_input_tokens"],
          decoder_positions=batch["decoder_positions"],
          decoder_segment_ids=batch["decoder_segment_ids"],
          mutable=list(state.keys()),
      )
      loss = _cross_entropy_loss(
          logits,
          batch["decoder_target_tokens"],
          batch["decoder_loss_weights"],
      )
      return loss, new_state

    (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, new_opt_state, loss

  data_iter = _make_data_iterator(config, tokenizer)

  for step in range(int(config.steps)):
    try:
      batch_np = next(data_iter)
    except StopIteration:
      max_logging.log("Dataset exhausted before reaching configured steps; stopping early.")
      break

    batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
    params, state, opt_state, loss = train_step(params, state, opt_state, batch)

    if step % max(1, int(getattr(config, "log_period", 10))) == 0:
      max_logging.log(f"step={step} loss={float(loss):.6f}")

  max_logging.log("SFT training finished.")


if __name__ == "__main__":
  app.run(main)
