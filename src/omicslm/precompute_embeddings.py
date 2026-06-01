"""Offline precomputation utility for Funomics and Geneformer embeddings."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable

import numpy as np


ArrayExtractor = Callable[[np.ndarray], np.ndarray]


def load_extractor(spec: str) -> ArrayExtractor:
  module_name, function_name = spec.split(":", maxsplit=1)
  module = importlib.import_module(module_name)
  extractor = getattr(module, function_name)
  if not callable(extractor):
    raise TypeError(f"Extractor {spec!r} must resolve to a callable.")
  return extractor


def batched_extract(expressions: np.ndarray, extractor: ArrayExtractor, batch_size: int) -> np.ndarray:
  outputs = []
  for start in range(0, expressions.shape[0], batch_size):
    outputs.append(np.asarray(extractor(expressions[start : start + batch_size])))
  return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


def precompute_embeddings(
    expressions: np.ndarray,
    funomics_extractor: ArrayExtractor,
    geneformer_extractor: ArrayExtractor,
    batch_size: int,
) -> dict[str, np.ndarray]:
  return {
      "funomics": batched_extract(expressions, funomics_extractor, batch_size).astype(np.float32),
      "geneformer": batched_extract(expressions, geneformer_extractor, batch_size).astype(np.float32),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--expressions", required=True, help="Path to a .npy or .npz expression array.")
  parser.add_argument("--output", required=True, help="Output .npz path.")
  parser.add_argument("--funomics-extractor", required=True, help="Callable spec module:function.")
  parser.add_argument("--geneformer-extractor", required=True, help="Callable spec module:function.")
  parser.add_argument("--batch-size", type=int, default=512)
  args = parser.parse_args()

  loaded = np.load(args.expressions)
  expressions = loaded["expressions"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
  outputs = precompute_embeddings(
      np.asarray(expressions),
      load_extractor(args.funomics_extractor),
      load_extractor(args.geneformer_extractor),
      args.batch_size,
  )
  Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.output, **outputs)


if __name__ == "__main__":
  main()
