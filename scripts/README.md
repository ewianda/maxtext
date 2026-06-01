# End-to-End Training Guide

This directory contains training scripts for both **MaxText** (the core JAX/TPU/GPU LLM pre-trainer) and **OmicsLM** (the multimodal instruction-tuning layer built on top of MaxText).

---

## Contents

| File | Purpose |
|---|---|
| `run_omicslm_train.py` | Standalone Python training script for OmicsLM with synthetic data |
| `run_omicslm_smoke.sh` | Shell wrapper — easiest way to launch an end-to-end OmicsLM run |
| `run_olmo3_7b_grain_smoke.sh` | MaxText pre-training smoke test for OLMo 3 7B on grain data |
| `run_olmo3_7b_grain_resume_test.sh` | Resume / checkpoint test for OLMo 3 7B |
| `run_qwen3_30b_rl.sh` | Reinforcement-learning (GRPO) script for Qwen3-30B |
| `run_qwen3_30b_hf_to_maxtext.sh` | Convert Qwen3-30B weights from HuggingFace → MaxText format |
| `run_qwen3_30b_maxtext_to_hf.sh` | Convert Qwen3-30B weights MaxText → HuggingFace format |

---

## 1  Installation

> Requires **Python 3.12** and either a CUDA 12 GPU or a Google Cloud TPU VM.

```bash
git clone https://github.com/<your-org>/maxtext.git
cd maxtext

# GPU (CUDA 12)
pip install -e ".[cuda12]"

# TPU
pip install -e ".[tpu]"
```

All scripts expect `src/` on the Python path.  When calling Python directly,
prefix with:
```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```
The shell wrappers do this automatically.

---

## 2  OmicsLM — end-to-end training

OmicsLM augments a decoder-only LLM with a learned omics projection layer.
The `<omics>` placeholder token is replaced in the embedding stream by a
projected omics vector before the backbone sees the sequence.

### 2.1  Quick smoke test (no external data required)

```bash
bash scripts/run_omicslm_smoke.sh
```

All data is synthetic.  Works on CPU, GPU, or TPU with no cloud credentials.
Expected output (50 steps):

```
=== OmicsLM smoke run ===
  steps         : 50
  batch_size    : 2
  ...
step     1/50  loss=6.9123  elapsed=3.4s
step    10/50  loss=6.8901  elapsed=5.1s
...
Training complete in 12.3s
Checkpoint saved to /tmp/omicslm_out/omicslm_params.npz
=== OmicsLM smoke run PASSED ===
```

### 2.2  Customising the run

All parameters can be overridden via environment variables (shell script) or
CLI flags (Python script):

```bash
# Larger model, longer run
STEPS=500 HIDDEN_SIZE=256 NUM_LAYERS=4 BATCH_SIZE=8 \
    bash scripts/run_omicslm_smoke.sh

# Or call the Python script directly for full control
python scripts/run_omicslm_train.py \
    --steps 1000 \
    --batch_size 8 \
    --seq_len 512 \
    --learning_rate 5e-5 \
    --hidden_size 4096 \
    --num_layers 32 \
    --omics_dim 21287 \
    --output_dir /path/to/output
```

Full list of Python CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--batch_size` | `2` | Training batch size |
| `--seq_len` | `64` | Token sequence length |
| `--steps` | `50` | Number of training steps |
| `--log_every` | `10` | Log loss every N steps |
| `--learning_rate` | `1e-4` | Adam learning rate |
| `--hidden_size` | `64` | LLM embedding/hidden dimension |
| `--num_layers` | `2` | Number of transformer layers |
| `--vocab_size` | `1024` | Vocabulary size |
| `--omics_dim` | `21287` | Omics input vector dimension (`1 + 20006 + 512 + 768`) |
| `--omics_token_id` | `32` | Token ID used as `<omics>` placeholder |
| `--output_dir` | `/tmp/omicslm_out` | Directory for saving the checkpoint |

### 2.3  Swapping in a real backbone

The script ships with a tiny smoke backbone.  To use a real MaxText or
HuggingFace backbone, edit `run_omicslm_train.py` and replace
`_TinyBackboneWithEmbedder` with your backbone class.  The only requirement
is that it exposes a `shared_embedding(input_ids)` method (or
`token_embedder`) and returns `{"logits": ...}` from `__call__`.

### 2.4  Using real omics data

Replace the `_data_iterator` call in `train()` with your own DataLoader that
yields dictionaries with keys:

| Key | Shape | dtype | Description |
|---|---|---|---|
| `input_ids` | `[B, T]` | `int32` | Token IDs; position(s) with `omics_token_id` are the `<omics>` placeholders |
| `omics_inputs` | `[B, N, omics_dim]` | `float32` | Projected omics vectors — one per placeholder |
| `labels` | `[B, T]` | `int32` | Same as `input_ids` but with `ignore_index=-100` for positions to exclude from the loss |

Build omics vectors from raw expression data using
`omicslm.data_pipeline.build_omics_vector` (see step-by-step example in the
[main README](../README.md)):

```python
from omicslm.data_pipeline import build_omics_vector
from omicslm.utils import load_norm_stats, load_gene_panel

gene_panel = load_gene_panel("path/to/gene_panel.txt")
stats = load_norm_stats("path/to/norm_stats.npz")

omics_vec = build_omics_vector(
    expression={"BRCA1": 1.2, "TP53": 0.8, ...},  # dict[gene_name, count]
    funomics_embedding=funomics_arr,                 # np.ndarray [512]
    geneformer_embedding=geneformer_arr,             # np.ndarray [768]
    stats=stats,
    sample_type="single_cell",                       # or "bulk"
)
# omics_vec.shape == (21287,)
```

---

## 3  MaxText — LLM pre-training

### 3.1  GPU smoke test (synthetic data, no GCS)

```bash
python -m maxtext.trainers.pre_train.train \
    src/maxtext/configs/gpu/gpu_smoke_test.yml \
    run_name=smoke_$(date +%Y%m%d) \
    base_output_directory=/tmp/maxtext_out
```

### 3.2  GPU — Llama-2-7B on a single 8-GPU node

```bash
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
  --xla_gpu_enable_triton_gemm=false \
  --xla_gpu_enable_command_buffer='' \
  --xla_gpu_all_reduce_combine_threshold_bytes=134217728"

python -m maxtext.trainers.pre_train.train \
    src/maxtext/configs/models/gpu/llama2_7b.yml \
    run_name=llama2_7b_$(date +%Y%m%d) \
    dcn_data_parallelism=1 \
    ici_fsdp_parallelism=8 \
    base_output_directory=/tmp/maxtext_out \
    dataset_type=synthetic
```

### 3.3  TPU — Llama-2-7B synthetic run

```bash
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 \
  --xla_enable_async_all_gather=true"

python -m maxtext.trainers.pre_train.train \
    src/maxtext/configs/base.yml \
    model_name=llama2-7b \
    run_name=llama2_tpu_$(date +%Y%m%d) \
    steps=15 \
    per_device_batch_size=12 \
    enable_checkpointing=false \
    remat_policy=full \
    ici_fsdp_parallelism=-1 \
    max_target_length=4096 \
    base_output_directory=/tmp/maxtext_out \
    dataset_type=synthetic \
    attention=flash
```

Any model from `src/maxtext/configs/models/*.yml` can be substituted by
changing `model_name=` (e.g. `gemma3-4b`, `qwen3-8b`, `deepseek3-tiny`).

### 3.4  Real dataset

Replace `dataset_type=synthetic` with `dataset_type=tfds` and add
`dataset_path=gs://<your-bucket>/` pointing at a preprocessed TF-Records
dataset, or use `dataset_type=hf` with `hf_train_files=<path>` for a local
HuggingFace dataset.

---

## 4  Checkpoint loading

After training, load the OmicsLM parameter checkpoint:

```python
import numpy as np
data = np.load("/tmp/omicslm_out/omicslm_params.npz")
# Reconstruct a pytree from the flat key paths if needed.
```

For MaxText, point `load_parameters_path=` at the Orbax checkpoint directory
(see `src/maxtext/configs/base.yml` for all checkpointing options).

---

## 5  Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: omicslm` | Run `export PYTHONPATH=$PWD/src:$PYTHONPATH` |
| `jax.errors.UnexpectedTracerError` | Ensure `input_ids` are passed as `jnp.asarray(...)` |
| `ValueError: Each sequence must provide exactly one projected omics vector` | Number of `omics_token_id` tokens in `input_ids` must equal `omics_inputs.shape[1]` per sequence |
| CUDA OOM | Reduce `--batch_size`, `--seq_len`, or `--hidden_size` |
| TPU `ResourceExhausted` | Reduce `per_device_batch_size` or enable `remat_policy=full` |
