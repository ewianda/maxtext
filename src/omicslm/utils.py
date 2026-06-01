"""Shared OmicsLM utility helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from omicslm.data_pipeline import OmicsNormalizationStats


def load_gene_panel(path: str | Path) -> list[str]:
  return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_norm_stats(path: str | Path, stats: OmicsNormalizationStats) -> None:
  np.savez(
      path,
      gene_means=stats.gene_means,
      expression_std=np.asarray(stats.expression_std, dtype=np.float32),
      scale_mean=np.asarray(stats.scale_mean, dtype=np.float32),
      scale_std=np.asarray(stats.scale_std, dtype=np.float32),
      funomics_mean=np.asarray(stats.funomics_mean if stats.funomics_mean is not None else [], dtype=np.float32),
      funomics_std=np.asarray(stats.funomics_std if stats.funomics_std is not None else [], dtype=np.float32),
      geneformer_mean=np.asarray(stats.geneformer_mean if stats.geneformer_mean is not None else [], dtype=np.float32),
      geneformer_std=np.asarray(stats.geneformer_std if stats.geneformer_std is not None else [], dtype=np.float32),
  )


def load_norm_stats(path: str | Path) -> OmicsNormalizationStats:
  data = np.load(path)

  def _maybe_array(name: str):
    value = np.asarray(data[name])
    return None if value.size == 0 else value.astype(np.float32)

  return OmicsNormalizationStats(
      gene_means=np.asarray(data["gene_means"], dtype=np.float32),
      expression_std=float(np.asarray(data["expression_std"])),
      scale_mean=float(np.asarray(data["scale_mean"])),
      scale_std=float(np.asarray(data["scale_std"])),
      funomics_mean=_maybe_array("funomics_mean"),
      funomics_std=_maybe_array("funomics_std"),
      geneformer_mean=_maybe_array("geneformer_mean"),
      geneformer_std=_maybe_array("geneformer_std"),
  )
