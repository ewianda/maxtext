"""Utilities for building GEO-OmicsQA style grounded QA datasets."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from omicslm.omics_embedding import OMICS_PLACEHOLDER


@dataclass(frozen=True)
class GeoOmicsQAExample:
  question: str
  answer: str
  omics_profiles: tuple[str, ...]
  source_id: str


def render_geo_omics_qa(example: GeoOmicsQAExample) -> dict[str, str]:
  question = example.question
  for profile in example.omics_profiles:
    question = question.replace(profile, OMICS_PLACEHOLDER, 1)
  return {
      "source_id": example.source_id,
      "question": question,
      "answer": example.answer,
  }


def split_examples(
    examples: Iterable[GeoOmicsQAExample],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, list[GeoOmicsQAExample]]:
  examples = list(examples)
  rng = random.Random(seed)
  rng.shuffle(examples)
  train_end = int(len(examples) * train_fraction)
  val_end = train_end + int(len(examples) * val_fraction)
  return {
      "train": examples[:train_end],
      "val": examples[train_end:val_end],
      "test": examples[val_end:],
  }
