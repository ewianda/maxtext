"""Instruction-tuning data generation helpers for OmicsLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import random

from omicslm.omics_embedding import OMICS_PLACEHOLDER


@dataclass(frozen=True)
class InstructionExample:
  dataset: str
  system: str
  user_template: str
  assistant: str
  omics_references: tuple[str, ...] = ()
  metadata: Mapping[str, str] | None = None


def render_instruction_example(example: InstructionExample) -> dict[str, list[dict[str, str]]]:
  prompt = example.user_template
  for reference in example.omics_references:
    prompt = prompt.replace(reference, OMICS_PLACEHOLDER, 1)
  return {
      "messages": [
          {"role": "system", "content": example.system},
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": example.assistant},
      ]
  }


def subsample_examples(
    examples: Sequence[InstructionExample],
    max_per_dataset: Mapping[str, int],
    seed: int = 0,
) -> list[InstructionExample]:
  rng = random.Random(seed)
  grouped: dict[str, list[InstructionExample]] = {}
  for example in examples:
    grouped.setdefault(example.dataset, []).append(example)
  sampled: list[InstructionExample] = []
  for dataset, dataset_examples in grouped.items():
    limit = max_per_dataset.get(dataset, len(dataset_examples))
    if len(dataset_examples) <= limit:
      sampled.extend(dataset_examples)
    else:
      sampled.extend(rng.sample(dataset_examples, limit))
  return sampled


def render_corpus(examples: Iterable[InstructionExample]) -> list[dict[str, list[dict[str, str]]]]:
  return [render_instruction_example(example) for example in examples]
