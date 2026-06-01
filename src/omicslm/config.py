"""Configuration helpers for OmicsLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class OmicsLMConfig:
  """Configuration for OmicsLM-specific components."""

  omics_dim: int = 20541
  expression_dim: int = 19260
  funomics_dim: int = 512
  geneformer_dim: int = 768
  input_scale_dim: int = 1
  projection_init_gain: float = 0.01
  omics_token: str = "<omics>"
  omics_token_id: int | None = None
  num_gene_tokens: int = 19260
  gene_panel_path: str | None = None
  norm_stats_path: str | None = None
  hidden_size: int = 4096
  vocab_size: int | None = None
  learning_rate: float = 1e-5
  batch_size: int = 1
  data_mix_weights: Mapping[str, float] = field(default_factory=dict)
  backbone_config: Mapping[str, Any] = field(default_factory=dict)

  @property
  def expected_omics_dim(self) -> int:
    return self.input_scale_dim + self.expression_dim + self.funomics_dim + self.geneformer_dim
