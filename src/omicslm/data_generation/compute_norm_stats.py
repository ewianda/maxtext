"""Compute OmicsLM normalization statistics from training features."""

from __future__ import annotations

import numpy as np

from omicslm.data_pipeline import OmicsNormalizationStats


def compute_norm_stats(
    expression_matrix: np.ndarray,
    scale_indicators: np.ndarray,
    funomics_embeddings: np.ndarray,
    geneformer_embeddings: np.ndarray,
) -> OmicsNormalizationStats:
  expression_matrix = np.asarray(expression_matrix, dtype=np.float32)
  logged_expression = np.log1p(expression_matrix)
  return OmicsNormalizationStats(
      gene_means=logged_expression.mean(axis=0),
      expression_std=float(np.maximum(logged_expression.std(), 1e-8)),
      scale_mean=float(np.asarray(scale_indicators, dtype=np.float32).mean()),
      scale_std=float(np.maximum(np.asarray(scale_indicators, dtype=np.float32).std(), 1e-8)),
      funomics_mean=np.asarray(funomics_embeddings, dtype=np.float32).mean(axis=0),
      funomics_std=np.maximum(np.asarray(funomics_embeddings, dtype=np.float32).std(axis=0), 1e-8),
      geneformer_mean=np.asarray(geneformer_embeddings, dtype=np.float32).mean(axis=0),
      geneformer_std=np.maximum(np.asarray(geneformer_embeddings, dtype=np.float32).std(axis=0), 1e-8),
  )
