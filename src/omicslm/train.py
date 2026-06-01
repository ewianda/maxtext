"""Training helpers for OmicsLM instruction tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def normalize_sampling_weights(weights: Mapping[str, float]) -> dict[str, float]:
  total = float(sum(weights.values()))
  if total <= 0.0:
    raise ValueError("Sampling weights must sum to a positive value.")
  return {name: value / total for name, value in weights.items()}


def build_batch_loss_weights(
    input_ids: np.ndarray,
    labels: np.ndarray,
    omics_token_id: int,
    ignore_index: int = -100,
) -> np.ndarray:
  weights = np.ones_like(labels, dtype=np.float32)
  weights *= labels != ignore_index
  weights *= input_ids != omics_token_id
  return weights


def collate_batch(examples: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
  if not examples:
    raise ValueError("Cannot collate an empty batch.")
  collated: dict[str, np.ndarray] = {}
  for key in examples[0]:
    collated[key] = np.stack([np.asarray(example[key]) for example in examples], axis=0)
  return collated
