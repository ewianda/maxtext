"""OmicsLM model wrapper for decoder-only language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from omicslm.omics_embedding import inject_omics_embeddings
from omicslm.omics_projection import OmicsProjection


@dataclass(frozen=True)
class OmicsLMOutput:
  logits: jnp.ndarray
  projected_omics: jnp.ndarray | None
  input_embeddings: jnp.ndarray
  loss: jnp.ndarray | None
  loss_weights: jnp.ndarray | None
  backbone_outputs: Any


class OmicsLM(nn.Module):
  """Injects projected omics vectors into a decoder-only LLM token stream."""

  backbone: nn.Module
  omics_dim: int
  hidden_size: int
  omics_token_id: int
  projection_init_gain: float = 0.01
  token_embedder: nn.Module | None = None
  ignore_index: int = -100

  def setup(self):
    self.omics_projection = OmicsProjection(
        input_dim=self.omics_dim,
        hidden_size=self.hidden_size,
        gain=self.projection_init_gain,
    )

  def _embed_tokens(self, input_ids: jnp.ndarray) -> jnp.ndarray:
    if self.token_embedder is not None:
      return self.token_embedder(input_ids.astype(jnp.int32))
    if hasattr(self.backbone, "shared_embedding"):
      return self.backbone.shared_embedding(input_ids.astype(jnp.int32))
    if hasattr(self.backbone, "token_embedder"):
      return self.backbone.token_embedder(input_ids.astype(jnp.int32))
    raise ValueError("OmicsLM requires either token_embedder or a backbone with shared_embedding/token_embedder.")

  def build_loss_weights(
      self,
      input_ids: jnp.ndarray,
      labels: jnp.ndarray | None,
      loss_weights: jnp.ndarray | None = None,
  ) -> jnp.ndarray | None:
    if labels is None:
      return None
    weights = jnp.ones_like(labels, dtype=jnp.float32) if loss_weights is None else jnp.asarray(loss_weights, dtype=jnp.float32)
    weights = weights * (labels != self.ignore_index)
    weights = weights * (input_ids != self.omics_token_id)
    return weights

  def compute_causal_lm_loss(
      self,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      loss_weights: jnp.ndarray,
  ) -> jnp.ndarray:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_weights = loss_weights[:, 1:]
    safe_labels = jnp.clip(shifted_labels, 0, shifted_logits.shape[-1] - 1)
    token_log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
    gathered = jnp.take_along_axis(token_log_probs, safe_labels[..., None], axis=-1).squeeze(axis=-1)
    per_token_loss = -gathered * shifted_weights
    normalizer = jnp.maximum(shifted_weights.sum(), 1.0)
    return per_token_loss.sum() / normalizer

  def __call__(
      self,
      input_ids: jnp.ndarray,
      *,
      omics_inputs: jnp.ndarray | None = None,
      omics_mask: jnp.ndarray | None = None,
      labels: jnp.ndarray | None = None,
      loss_weights: jnp.ndarray | None = None,
      inputs_embeds: jnp.ndarray | None = None,
      **backbone_kwargs,
  ) -> OmicsLMOutput:
    input_ids = jnp.asarray(input_ids)
    input_embeddings = self._embed_tokens(input_ids) if inputs_embeds is None else jnp.asarray(inputs_embeds)

    projected_omics = None
    if omics_inputs is not None:
      projected_omics = self.omics_projection(jnp.asarray(omics_inputs))
      input_embeddings = inject_omics_embeddings(
          input_embeddings,
          input_ids,
          projected_omics,
          self.omics_token_id,
          omics_mask=omics_mask,
      )

    backbone_outputs = self.backbone(input_ids=input_ids, inputs_embeds=input_embeddings, **backbone_kwargs)
    if isinstance(backbone_outputs, dict):
      logits = backbone_outputs["logits"]
    else:
      logits = getattr(backbone_outputs, "logits", backbone_outputs)

    effective_loss_weights = self.build_loss_weights(input_ids, labels, loss_weights)
    loss = None
    if labels is not None and effective_loss_weights is not None:
      loss = self.compute_causal_lm_loss(jnp.asarray(logits), jnp.asarray(labels), effective_loss_weights)

    return OmicsLMOutput(
        logits=jnp.asarray(logits),
        projected_omics=projected_omics,
        input_embeddings=input_embeddings,
        loss=loss,
        loss_weights=effective_loss_weights,
        backbone_outputs=backbone_outputs,
    )
