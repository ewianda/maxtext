from __future__ import annotations

import numpy as np

from omicslm.data_pipeline import (
    OmicsNormalizationStats,
    align_expression_profile,
    build_omics_vector,
    library_size_normalize,
    normalize_expression,
)


def test_align_expression_profile_fills_missing_genes_with_zero():
  aligned = align_expression_profile({"GENE2": 4.0, "GENE1": 2.0}, ["GENE1", "GENE3", "GENE2"])
  np.testing.assert_array_equal(aligned, np.array([2.0, 0.0, 4.0], dtype=np.float32))


def test_normalize_expression_applies_library_size_log_and_scaling():
  expression = np.array([1.0, 3.0], dtype=np.float32)
  normalized = normalize_expression(expression, gene_means=np.array([0.0, 0.0], dtype=np.float32), expression_std=2.0)
  expected = np.log1p(library_size_normalize(expression)) / 2.0
  np.testing.assert_allclose(normalized, expected)


def test_build_omics_vector_concatenates_all_feature_groups():
  stats = OmicsNormalizationStats(
      gene_means=np.array([0.5, 1.5], dtype=np.float32),
      expression_std=2.0,
      scale_mean=0.5,
      scale_std=0.5,
      funomics_mean=np.array([1.0, 2.0], dtype=np.float32),
      funomics_std=np.array([2.0, 4.0], dtype=np.float32),
      geneformer_mean=np.array([2.0, 4.0], dtype=np.float32),
      geneformer_std=np.array([2.0, 2.0], dtype=np.float32),
  )

  vector = build_omics_vector(
      expression=np.array([1.0, 3.0], dtype=np.float32),
      funomics_embedding=np.array([3.0, 10.0], dtype=np.float32),
      geneformer_embedding=np.array([4.0, 10.0], dtype=np.float32),
      stats=stats,
      sample_type="single_cell",
  )

  assert vector.shape == (1 + 2 + 2 + 2,)
  np.testing.assert_allclose(vector[0], 1.0)
