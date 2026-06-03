#!/usr/bin/env python3
"""
Ray Data pipeline: compute Geneformer + scGPT embeddings for OmicsLM ArrayRecords.

Reads ArrayRecords from GCS, extracts expression vectors, computes cell embeddings
via Geneformer V2 (1152-dim) and scGPT (512-dim), writes new ArrayRecords with
the full omics vector: [1 scale, 20006 expression, 512 scGPT, 1152 Geneformer].

GCS ArrayRecords -> CPU parse -> GPU/TPU encode (Geneformer + scGPT) -> GCS ArrayRecords

Usage:
    # Local test (GPU)
    python scripts/compute_gene_embeddings.py \
        --input_path /tmp/omicslm_shard0.array_record \
        --output_path /tmp/omicslm_with_embeddings/ \
        --norm_stats /tmp/omics_norm_stats.npz \
        --sample 100

    # Full run (Ray cluster)
    python scripts/compute_gene_embeddings.py \
        --input_path gs://omicslm-batch-data/arrayrecord/omicslm-*.array_record \
        --output_path gs://omicslm-batch-data/arrayrecord_v2/ \
        --norm_stats gs://omicslm-batch-data/omics_norm_stats.npz \
        --num_gpu_workers 4
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import types

# Bypass torchtext (scGPT dependency, binary incompatible with torch 2.7)
fake_torchtext = types.ModuleType("torchtext")
fake_torchtext._extension = types.ModuleType("_extension")
fake_vocab = types.ModuleType("vocab")
class _FakeVocab:
    def __init__(self, *a, **kw): pass
fake_vocab.Vocab = _FakeVocab
sys.modules["torchtext"] = fake_torchtext
sys.modules["torchtext._extension"] = fake_torchtext._extension
sys.modules["torchtext.vocab"] = fake_vocab

import numpy as np
import pyarrow as pa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gene-embeddings")


class ExpressionParser:
    """CPU: Parse ArrayRecords, extract expression vectors and metadata."""

    def __init__(self, norm_stats_path: str | None = None):
        self.gene_mean = None
        self.global_std = 1.0
        if norm_stats_path:
            if norm_stats_path.startswith("gs://"):
                import tensorflow as tf
                raw = tf.io.gfile.GFile(norm_stats_path, "rb").read()
                stats = np.load(io.BytesIO(raw))
            else:
                stats = np.load(norm_stats_path)
            self.gene_mean = stats["gene_mean"].astype(np.float32)
            self.global_std = float(stats["global_std"])

    def __call__(self, batch):
        import tensorflow as tf

        records = batch["bytes"] if "bytes" in batch else batch["value"]
        rows = []
        for raw_bytes in records:
            if hasattr(raw_bytes, "as_py"):
                raw_bytes = raw_bytes.as_py()
            example = tf.train.Example()
            example.ParseFromString(raw_bytes)
            f = example.features.feature

            prompt = f["prompt"].bytes_list.value[0].decode()
            completion = f["completion"].bytes_list.value[0].decode()

            omics_key = "omics" if "omics" in f else "omics_inputs"
            feat = f[omics_key]
            if feat.bytes_list.value:
                expr = np.frombuffer(feat.bytes_list.value[0], dtype=np.float32).copy()
            elif feat.float_list.value:
                expr = np.array(feat.float_list.value, dtype=np.float32)
            else:
                expr = np.zeros(20006, dtype=np.float32)

            rows.append({
                "prompt": prompt,
                "completion": completion,
                "expression_raw": expr.tobytes(),
                "expression_len": len(expr),
            })

        return pa.table({
            "prompt": [r["prompt"] for r in rows],
            "completion": [r["completion"] for r in rows],
            "expression_raw": [r["expression_raw"] for r in rows],
            "expression_len": [r["expression_len"] for r in rows],
        })


class GeneEmbedder:
    """GPU/TPU: Compute Geneformer + scGPT embeddings from expression vectors."""

    def __init__(self, geneformer_model: str = "ctheodoris/Geneformer",
                 scgpt_model: str = "tdc/scGPT",
                 norm_stats_path: str | None = None):
        import torch
        from transformers import AutoModel
        from scgpt.model.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        from huggingface_hub import hf_hub_download

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch

        # Load norm stats
        self.gene_mean = None
        self.global_std = 1.0
        if norm_stats_path:
            if norm_stats_path.startswith("gs://"):
                import tensorflow as tf
                raw = tf.io.gfile.GFile(norm_stats_path, "rb").read()
                stats = np.load(io.BytesIO(raw))
            else:
                stats = np.load(norm_stats_path)
            self.gene_mean = stats["gene_mean"].astype(np.float32)
            self.global_std = float(stats["global_std"])

        # Load Geneformer
        logger.info("Loading Geneformer from %s", geneformer_model)
        self.geneformer = AutoModel.from_pretrained(geneformer_model, trust_remote_code=True)
        self.geneformer.to(self.device).eval()
        self.gf_hidden = self.geneformer.config.hidden_size
        self.gf_vocab_size = self.geneformer.config.vocab_size
        logger.info("Geneformer: hidden=%d, vocab=%d", self.gf_hidden, self.gf_vocab_size)

        # Load scGPT
        logger.info("Loading scGPT vocab")
        vocab_file = hf_hub_download(scgpt_model, "vocab.json")
        self.scgpt_vocab = GeneVocab.from_file(vocab_file)

        # Load pretrained scGPT weights
        logger.info("Loading scGPT model")
        model_file = hf_hub_download(scgpt_model, "best_model.pt")
        self.scgpt = TransformerModel(
            ntoken=len(self.scgpt_vocab),
            d_model=512,
            nhead=8,
            d_hid=512,
            nlayers=12,
        )
        state = self.torch.load(model_file, map_location="cpu", weights_only=True)
        self.scgpt.load_state_dict(state, strict=False)
        self.scgpt.to(self.device).eval()
        logger.info("scGPT: d_model=512, vocab=%d", len(self.scgpt_vocab))
        logger.info("GeneEmbedder ready on %s", self.device)

    def _expression_to_geneformer_tokens(self, expression: np.ndarray) -> np.ndarray:
        """Convert expression vector to Geneformer rank-value encoding."""
        # Geneformer ranks genes by expression (highest first)
        # Token IDs correspond to gene indices in the vocab
        nonzero_mask = expression > 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_values = expression[nonzero_indices]

        # Rank by expression (descending)
        rank_order = np.argsort(-nonzero_values)
        ranked_gene_indices = nonzero_indices[rank_order]

        # Clip to vocab size and max sequence length
        ranked_gene_indices = ranked_gene_indices[ranked_gene_indices < self.gf_vocab_size]
        ranked_gene_indices = ranked_gene_indices[:2048]  # max seq len

        return ranked_gene_indices.astype(np.int64)

    def _expression_to_scgpt_input(self, expression: np.ndarray):
        """Convert expression to scGPT gene_ids + values."""
        nonzero_mask = expression > 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_values = expression[nonzero_indices]

        # Map to scGPT vocab (gene indices)
        # scGPT uses gene names -> vocab IDs, but we have indices
        # Use indices directly as token IDs (clipped to vocab)
        vocab_size = len(self.scgpt_vocab)
        valid = nonzero_indices < vocab_size
        gene_ids = nonzero_indices[valid][:2048]
        values = nonzero_values[valid][:2048]

        return gene_ids.astype(np.int64), values.astype(np.float32)

    def __call__(self, batch):
        records = batch.to_pylist() if hasattr(batch, "to_pylist") else [
            dict(zip(batch.keys(), vals)) for vals in zip(*batch.values())
        ]

        gf_embeddings = []
        scgpt_embeddings = []

        for rec in records:
            expr = np.frombuffer(rec["expression_raw"], dtype=np.float32).copy()

            # Geneformer: rank-value encoding -> BERT forward -> mean pool
            gf_tokens = self._expression_to_geneformer_tokens(expr)
            if len(gf_tokens) < 10:
                gf_tokens = np.zeros(10, dtype=np.int64)
            gf_input = self.torch.tensor(gf_tokens, dtype=self.torch.long).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                gf_out = self.geneformer(gf_input)
            gf_emb = gf_out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            gf_embeddings.append(gf_emb)

            # scGPT: gene_ids + values -> transformer -> mean pool
            gene_ids, values = self._expression_to_scgpt_input(expr)
            if len(gene_ids) < 10:
                gene_ids = np.zeros(10, dtype=np.int64)
                values = np.zeros(10, dtype=np.float32)
            gid_tensor = self.torch.tensor(gene_ids, dtype=self.torch.long).unsqueeze(0).to(self.device)
            val_tensor = self.torch.tensor(values, dtype=self.torch.float32).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                scgpt_out = self.scgpt(gid_tensor, val_tensor)
            sc_emb = scgpt_out.mean(dim=1).squeeze(0).cpu().numpy()
            scgpt_embeddings.append(sc_emb)

        # Build output: normalize expression + concatenate all features
        all_omics = []
        for i, rec in enumerate(records):
            expr = np.frombuffer(rec["expression_raw"], dtype=np.float32).copy()

            # Normalize expression
            if self.gene_mean is not None:
                expr_norm = (np.log1p(expr) - self.gene_mean[:len(expr)]) / max(self.global_std, 1e-8)
            else:
                expr_norm = np.log1p(expr)

            # Scale indicator (0=bulk, 1=single-cell, default 0.5=unknown)
            scale = np.array([0.5], dtype=np.float32)

            # Concatenate: [1 scale, N expression, 512 scGPT, 1152 Geneformer]
            omics_vec = np.concatenate([
                scale,
                expr_norm,
                scgpt_embeddings[i][:512].astype(np.float32),
                gf_embeddings[i][:1152].astype(np.float32),
            ])
            all_omics.append(omics_vec.tobytes())

        return pa.table({
            "prompt": [r["prompt"] for r in records],
            "completion": [r["completion"] for r in records],
            "omics": all_omics,
            "omics_dim": [len(np.frombuffer(o, dtype=np.float32)) for o in all_omics],
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--norm_stats", default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--num_gpu_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    import ray
    import ray.data as rd

    ray.init(runtime_env={"env_vars": {"HF_TOKEN": os.environ.get("HF_TOKEN", "")}})

    logger.info("Reading from %s", args.input_path)

    if args.input_path.endswith(".array_record"):
        from array_record.python.array_record_module import ArrayRecordReader
        reader = ArrayRecordReader(args.input_path)
        n = args.sample or reader.num_records()
        records = reader.read(list(range(n)))
        ds = rd.from_items([{"bytes": r} for r in records])
    else:
        ds = rd.read_binary_files(args.input_path)

    if args.sample:
        ds = ds.limit(args.sample)

    logger.info("Stage 1: CPU expression parsing...")
    ds = ds.map_batches(
        ExpressionParser,
        fn_constructor_kwargs={"norm_stats_path": args.norm_stats},
        batch_size=args.batch_size,
        num_cpus=1,
        concurrency=4,
    )

    logger.info("Stage 2: GPU gene embedding computation...")
    ds = ds.map_batches(
        GeneEmbedder,
        fn_constructor_kwargs={
            "norm_stats_path": args.norm_stats,
        },
        batch_size=8,
        num_gpus=1,
        concurrency=args.num_gpu_workers,
    )

    logger.info("Stage 3: Writing output to %s", args.output_path)
    ds.write_parquet(args.output_path)

    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
