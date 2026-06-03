#!/usr/bin/env bash
# OmicsLM 8B training on TPU v5litepod-16 (preemptible, resumable).
#
# Creates TPU, sets up env, converts weights (if needed), and trains.
# Checkpoints every 500 steps to GCS for preemption recovery.
# TensorBoard enabled.
#
# Usage:
#   bash scripts/tpu_train_8b.sh
#
# To resume after preemption:
#   bash scripts/tpu_train_8b.sh resume

set -euo pipefail

PROJECT="b6i-sandbox-discovery-ufg"
ZONE="us-west1-c"
TPU_NAME="omicslm-8b"
TPU_TYPE="v5litepod-16"
GCS_BASE="gs://omicslm-batch-data"
RUN_NAME="omicslm_8b_v1"

CHECKPOINT_DIR="${GCS_BASE}/omicslm_maxtext/${RUN_NAME}"
QWEN_CKPT="${GCS_BASE}/maxtext_checkpoints/qwen3-8b"
TRAIN_DATA="${GCS_BASE}/arrayrecord/omicslm-*.array_record"
NORM_STATS="${GCS_BASE}/omics_norm_stats.npz"
TOKENIZER="src/maxtext/assets/tokenizers/qwen3-tokenizer"

MODE="${1:-full}"

echo "=== OmicsLM 8B Training ==="
echo "Mode: ${MODE}"
echo "TPU: ${TPU_NAME} (${TPU_TYPE})"
echo "Run: ${RUN_NAME}"

# Step 1: Create TPU if not exists
if ! gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone="${ZONE}" --project="${PROJECT}" &>/dev/null; then
    echo "Creating TPU..."
    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
        --zone="${ZONE}" \
        --accelerator-type="${TPU_TYPE}" \
        --version=v2-alpha-tpuv5-lite \
        --project="${PROJECT}" \
        --preemptible
fi

# Step 2: Setup environment on TPU
echo "Setting up environment..."
gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --tunnel-through-iap \
    --worker=all \
    --command='
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

REPO_DIR="/tmp/maxtext-omicslm"
if [ ! -d "$REPO_DIR" ]; then
    git clone -b local/omicslm-test https://github.com/ewianda/maxtext.git "$REPO_DIR"
else
    cd "$REPO_DIR" && git pull
fi
cd "$REPO_DIR"

if [ ! -d ".venv" ]; then
    uv venv --python 3.12 .venv
    source .venv/bin/activate
    uv pip install -e "."
    uv pip install -r src/dependencies/requirements/generated_requirements/tpu-requirements.txt
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu safetensors
else
    source .venv/bin/activate
fi

python3 -c "import jax; print(f\"JAX: {len(jax.devices())} x {jax.devices()[0].platform}\")"
echo "=== ENV READY ==="
'

# Step 3: Convert Qwen3-8B weights if not already on GCS
if ! gsutil ls "${QWEN_CKPT}/0/items/" &>/dev/null; then
    echo "Converting Qwen3-8B weights..."
    gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
        --zone="${ZONE}" \
        --project="${PROJECT}" \
        --tunnel-through-iap \
        --command="
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd /tmp/maxtext-omicslm && source .venv/bin/activate
python3 -m maxtext.checkpoint_conversion.to_maxtext \
    src/maxtext/configs/base.yml \
    model_name=qwen3-8b \
    base_output_directory=${QWEN_CKPT} \
    scan_layers=True \
    hardware=cpu \
    skip_jax_distributed_system=True \
    checkpoint_storage_use_zarr3=True \
    checkpoint_storage_use_ocdbt=True \
    --eager_load_method=safetensors \
    --save_dtype=bfloat16
echo '=== CONVERSION DONE ==='
"
else
    echo "Qwen3-8B weights already on GCS, skipping conversion"
fi

# Step 4: Train
echo "Starting training..."
LOAD_PATH="${QWEN_CKPT}/0/items"
if [ "${MODE}" = "resume" ]; then
    LOAD_PATH=""
    echo "Resuming from latest checkpoint in ${CHECKPOINT_DIR}"
fi

gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --tunnel-through-iap \
    --command="
export PATH=\"\$HOME/.local/bin:\$PATH\"
cd /tmp/maxtext-omicslm && source .venv/bin/activate

python3 -m maxtext.trainers.pre_train.train \
    src/maxtext/configs/base.yml \
    model_name=qwen3-8b \
    run_name=${RUN_NAME} \
    base_output_directory=${GCS_BASE}/omicslm_maxtext \
    load_parameters_path=${LOAD_PATH} \
    grain_train_files='${TRAIN_DATA}' \
    tokenizer_path=${TOKENIZER} \
    omics_norm_stats_path=${NORM_STATS} \
    use_omics=true \
    omics_dim=20008 \
    omics_token_id=151669 \
    use_sft=true \
    sft_train_on_completion_only=true \
    packing=false \
    tokenize_train_data=true \
    'train_data_columns=[\"prompt\",\"completion\"]' \
    grain_file_type=arrayrecord \
    dataset_type=grain \
    steps=20000 \
    per_device_batch_size=4 \
    max_target_length=64 \
    enable_checkpointing=true \
    checkpoint_period=500 \
    attention=flash \
    scan_layers=True \
    remat_policy=full \
    gradient_clipping_threshold=1.0 \
    learning_rate=5e-5 \
    warmup_steps_fraction=0.02 \
    enable_tensorboard=true \
    log_period=10 \
    eval_interval=1000 \
    eval_steps=50 \
    'eval_data_columns=[\"prompt\",\"completion\"]' \
    grain_eval_files='${GCS_BASE}/arrayrecord/omicslm-00015-of-00016.array_record'
"

echo "=== TRAINING COMPLETE ==="
