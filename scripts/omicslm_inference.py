#!/usr/bin/env python3
"""Standalone OmicsLM inference — pure JAX, no JetStream dependency.

Loads a trained OmicsLM Orbax checkpoint and runs greedy autoregressive
decoding with omics vector injection.

Usage:
    python scripts/omicslm_inference.py \
        src/maxtext/configs/models/omicslm-qwen3-0.6b.yml \
        load_parameters_path=/tmp/omicslm_test_out/omicslm_2k/checkpoints/1999/items \
        tokenizer_path=src/maxtext/assets/tokenizers/qwen3-tokenizer \
        max_prefill_predict_length=128 \
        max_target_length=192 \
        per_device_batch_size=1 \
        scan_layers=True \
        hardware=gpu \
        skip_jax_distributed_system=True \
        attention=dot_product \
        enable_checkpointing=true \
        --omics_path=/tmp/omicslm_shard0.array_record \
        --omics_index=0 \
        --omics_norm_stats=/tmp/omics_norm_stats.npz \
        --prompt="What lineage is this cell line? <omics>"
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning

from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.models import models
from maxtext.utils import maxtext_utils


def load_omics_vector(path: str, index: int, norm_stats_path: str | None = None) -> np.ndarray:
    from array_record.python.array_record_module import ArrayRecordReader
    import tensorflow as tf

    reader = ArrayRecordReader(path)
    records = reader.read([index])
    example = tf.train.Example()
    example.ParseFromString(records[0])
    f = example.features.feature
    omics_key = "omics" if "omics" in f else "omics_inputs"
    feat = f[omics_key]
    if feat.bytes_list.value:
        vec = np.frombuffer(feat.bytes_list.value[0], dtype=np.float32).copy()
    elif feat.float_list.value:
        vec = np.array(feat.float_list.value, dtype=np.float32)
    else:
        raise ValueError(f"Empty omics field at index {index}")

    if norm_stats_path:
        stats = np.load(norm_stats_path)
        vec = (np.log1p(vec) - stats["gene_mean"]) / max(float(stats["global_std"]), 1e-8)
    return vec


def load_prompt_from_record(path: str, index: int) -> tuple[str, str]:
    from array_record.python.array_record_module import ArrayRecordReader
    import tensorflow as tf

    reader = ArrayRecordReader(path)
    records = reader.read([index])
    example = tf.train.Example()
    example.ParseFromString(records[0])
    f = example.features.feature
    prompt = f["prompt"].bytes_list.value[0].decode()
    completion = f["completion"].bytes_list.value[0].decode()
    return prompt, completion


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--omics_path", type=str, default=None)
    parser.add_argument("--omics_index", type=int, default=0)
    parser.add_argument("--omics_norm_stats", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--use_record_prompt", action="store_true")
    our_args, maxtext_argv = parser.parse_known_args()

    config = pyconfig.initialize([sys.argv[0]] + maxtext_argv)

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    model = models.transformer_as_linen(
        config=config, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN,
    )

    # Load tokenizer
    from maxtext.input_pipeline import tokenizer as tok_module
    tokenizer = tok_module.build_tokenizer(
        config.tokenizer_path, config.add_bos, config.add_eos,
        hf_access_token=getattr(config, "hf_access_token", None),
    )

    # Load checkpoint — init model then restore params from Orbax
    rng = jax.random.PRNGKey(0)
    init_rng = {"params": rng, "aqt": rng, "dropout": rng}
    B, T = 1, config.max_target_length
    dummy_ids = jnp.zeros((B, T), dtype=jnp.int32)
    dummy_pos = jnp.zeros((B, T), dtype=jnp.int32)
    dummy_seg = jnp.zeros((B, T), dtype=jnp.int32)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        init_vars = model.init(init_rng, dummy_ids, dummy_pos, dummy_seg, enable_dropout=False)

    from orbax import checkpoint as ocp
    ckpt_path = config.load_parameters_path
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(ckpt_path, args=ocp.args.StandardRestore(init_vars["params"]))
    params = {"params": restored}
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model loaded. Params: {param_count:,}")

    # Load omics
    omics_inputs = None
    if our_args.omics_path:
        norm_path = our_args.omics_norm_stats or getattr(config, "omics_norm_stats_path", "") or None
        vec = load_omics_vector(our_args.omics_path, our_args.omics_index, norm_path)
        omics_inputs = jnp.array(vec, dtype=jnp.float32).reshape(1, 1, -1)
        print(f"Omics vector: shape={vec.shape}, range=[{vec.min():.3f}, {vec.max():.3f}]")

    # Get prompt
    if our_args.use_record_prompt and our_args.omics_path:
        prompt, expected = load_prompt_from_record(our_args.omics_path, our_args.omics_index)
        print(f"Record prompt: {prompt}")
        print(f"Expected answer: {expected}")
    else:
        prompt = our_args.prompt or "What lineage is this cell line? <omics>"
        expected = None

    # Tokenize
    input_ids = tokenizer.encode(prompt)
    if isinstance(input_ids, list):
        input_ids = np.array(input_ids, dtype=np.int32)
    prompt_len = len(input_ids)
    print(f"Prompt tokens: {prompt_len}")

    # Pad to max_prefill_predict_length
    max_len = config.max_target_length
    padded = np.zeros(max_len, dtype=np.int32)
    padded[:prompt_len] = input_ids[:max_len]
    input_ids_jax = jnp.array(padded).reshape(1, -1)
    positions = jnp.arange(max_len).reshape(1, -1)
    segment_ids = jnp.where(jnp.arange(max_len) < prompt_len, 1, 0).reshape(1, -1)

    # Forward pass — get logits for all positions at once
    print("Running prefill...")
    with jax.default_device(jax.devices()[0]):
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            logits = model.apply(
                params,
                input_ids_jax,
                positions,
                decoder_segment_ids=segment_ids,
                omics_inputs=omics_inputs,
                enable_dropout=False,
                model_mode=MODEL_MODE_TRAIN,
            )

    # Greedy decode from the last prompt position
    next_token = jnp.argmax(logits[0, prompt_len - 1, :]).item()
    generated = [next_token]
    print(f"First token: {next_token} = {tokenizer.decode([next_token])!r}")

    # Simple autoregressive loop (no KV cache — slow but works)
    current_ids = padded.copy()
    for step in range(our_args.max_new_tokens - 1):
        pos = prompt_len + step
        if pos >= max_len - 1:
            break
        current_ids[pos] = next_token
        current_jax = jnp.array(current_ids).reshape(1, -1)
        seg = jnp.where(jnp.arange(max_len) <= pos, 1, 0).reshape(1, -1)

        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            logits = model.apply(
                params,
                current_jax,
                positions,
                decoder_segment_ids=seg,
                omics_inputs=omics_inputs,
                enable_dropout=False,
                model_mode=MODEL_MODE_TRAIN,
            )

        next_token = jnp.argmax(logits[0, pos, :]).item()
        generated.append(next_token)

        eos_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None)
        if eos_id and next_token == eos_id:
            break

    output = tokenizer.decode(generated)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {output}")
    if expected:
        print(f"Expected: {expected}")


if __name__ == "__main__":
    main()
