"""Normalization and feature construction for OmicsLM inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


_EPS = 1e-8


@dataclass(frozen=True)
class OmicsNormalizationStats:
  gene_means: np.ndarray
  expression_std: float
  scale_mean: float = 0.0
  scale_std: float = 1.0
  funomics_mean: np.ndarray | None = None
  funomics_std: np.ndarray | None = None
  geneformer_mean: np.ndarray | None = None
  geneformer_std: np.ndarray | None = None


def align_expression_profile(
    expression_by_gene: Mapping[str, float],
    gene_panel: Sequence[str],
) -> np.ndarray:
  """Aligns a sparse expression mapping to the fixed gene panel order."""
  return np.asarray([expression_by_gene.get(gene, 0.0) for gene in gene_panel], dtype=np.float32)


def library_size_normalize(expression: np.ndarray, target_sum: float = 1e6) -> np.ndarray:
  expression = np.asarray(expression, dtype=np.float32)
  library_size = float(expression.sum())
  if library_size <= 0.0:
    return np.zeros_like(expression)
  return expression * (target_sum / library_size)


def normalize_expression(
    expression: np.ndarray,
    gene_means: np.ndarray,
    expression_std: float,
    *,
    assume_library_normalized: bool = False,
) -> np.ndarray:
  """Applies OmicsLM expression normalization."""
  expression = np.asarray(expression, dtype=np.float32)
  gene_means = np.asarray(gene_means, dtype=np.float32)
  if not assume_library_normalized:
    expression = library_size_normalize(expression)
  expression = np.log1p(expression)
  return (expression - gene_means) / max(float(expression_std), _EPS)


def standardize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
  features = np.asarray(features, dtype=np.float32)
  mean = np.asarray(mean, dtype=np.float32)
  std = np.asarray(std, dtype=np.float32)
  return (features - mean) / np.maximum(std, _EPS)


def sample_scale_indicator(sample_type: str) -> float:
  normalized = sample_type.strip().lower()
  if normalized in {"single_cell", "single-cell", "sc", "count", "counts"}:
    return 1.0
  if normalized in {"bulk", "tpm", "bulk_rna", "bulk-rna"}:
    return 0.0
  raise ValueError(f"Unsupported sample type: {sample_type!r}.")


def build_omics_vector(
    expression: np.ndarray,
    funomics_embedding: np.ndarray,
    geneformer_embedding: np.ndarray,
    stats: OmicsNormalizationStats,
    *,
    sample_type: str,
    assume_library_normalized: bool | None = None,
) -> np.ndarray:
  """Builds the full OmicsLM input vector by concatenating scale, expression, and pretrained embedding features."""
  if assume_library_normalized is None:
    assume_library_normalized = sample_scale_indicator(sample_type) == 0.0

  scale_feature = standardize_features(
      np.asarray([sample_scale_indicator(sample_type)], dtype=np.float32),
      np.asarray([stats.scale_mean], dtype=np.float32),
      np.asarray([stats.scale_std], dtype=np.float32),
  )
  normalized_expression = normalize_expression(
      expression,
      stats.gene_means,
      stats.expression_std,
      assume_library_normalized=assume_library_normalized,
  )
  normalized_funomics = standardize_features(
      funomics_embedding,
      np.zeros_like(funomics_embedding) if stats.funomics_mean is None else stats.funomics_mean,
      np.ones_like(funomics_embedding) if stats.funomics_std is None else stats.funomics_std,
  )
  normalized_geneformer = standardize_features(
      geneformer_embedding,
      np.zeros_like(geneformer_embedding) if stats.geneformer_mean is None else stats.geneformer_mean,
      np.ones_like(geneformer_embedding) if stats.geneformer_std is None else stats.geneformer_std,
  )
  return np.concatenate(
      [scale_feature, normalized_expression, normalized_funomics, normalized_geneformer], axis=0
  ).astype(np.float32)
