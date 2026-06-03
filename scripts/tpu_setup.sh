#!/usr/bin/env bash
# OmicsLM TPU setup — installs from the working branch with all fixes.
#
# Usage:
#   # Create TPU (run from your local machine or cloud shell):
#   gcloud compute tpus tpu-vm create omicslm-v5e-8 \
#       --zone=us-west1-c \
#       --accelerator-type=v5litepod-8 \
#       --version=v2-alpha-tpuv5-lite \
#       --project=b6i-sandbox-discovery-ufg \
#       --preemptible
#
#   # SSH and run this script:
#   gcloud compute tpus tpu-vm ssh omicslm-v5e-8 \
#       --zone=us-west1-c \
#       --project=b6i-sandbox-discovery-ufg \
#       -- 'bash -s' < scripts/tpu_setup.sh
#
#   # Or for multi-host (v5litepod-16+):
#   gcloud compute tpus tpu-vm ssh omicslm-v5e-16 \
#       --zone=us-west1-c \
#       --project=b6i-sandbox-discovery-ufg \
#       --worker=all \
#       -- 'bash -s' < scripts/tpu_setup.sh

set -euo pipefail

echo "=== OmicsLM TPU Setup ==="

# 1. Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Clone the repo with our fixes
REPO_DIR="/tmp/maxtext-omicslm"
if [ -d "$REPO_DIR" ]; then
    echo "Removing existing repo..."
    rm -rf "$REPO_DIR"
fi
echo "Cloning repo..."
git clone -b local/omicslm-test https://github.com/ewianda/maxtext.git "$REPO_DIR"
cd "$REPO_DIR"

# 3. Create venv with Python 3.12
echo "Creating venv..."
uv venv --python 3.12 .venv
source .venv/bin/activate

# 4. Install MaxText with TPU dependencies
echo "Installing MaxText + TPU deps..."
uv pip install -e "."
uv pip install -r src/dependencies/requirements/generated_requirements/tpu-requirements.txt

# 5. Install conversion deps
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install safetensors

# 6. Verify JAX sees TPU devices
echo "=== Verifying TPU ==="
python3 -c "
import jax
devices = jax.devices()
print(f'JAX devices: {len(devices)} x {devices[0].platform}')
for d in devices:
    print(f'  {d}')
"

echo ""
echo "=== Setup complete ==="
echo "Repo: $REPO_DIR"
echo "Venv: $REPO_DIR/.venv"
echo ""
echo "Next steps:"
echo "  source $REPO_DIR/.venv/bin/activate"
echo "  cd $REPO_DIR"
echo ""
echo "  # Convert Qwen3-4B weights:"
echo "  python3 -m maxtext.checkpoint_conversion.to_maxtext \\"
echo "      src/maxtext/configs/base.yml \\"
echo "      model_name=qwen3-4b \\"
echo "      base_output_directory=gs://omicslm-batch-data/maxtext_checkpoints/qwen3-4b \\"
echo "      scan_layers=True hardware=cpu skip_jax_distributed_system=True \\"
echo "      checkpoint_storage_use_zarr3=True checkpoint_storage_use_ocdbt=True \\"
echo "      --eager_load_method=safetensors --save_dtype=bfloat16"
echo ""
echo "  # Train OmicsLM 4B:"
echo "  python3 -m maxtext.trainers.pre_train.train \\"
echo "      src/maxtext/configs/models/omicslm-qwen3-4b.yml \\"
echo "      run_name=omicslm_4b \\"
echo "      base_output_directory=gs://omicslm-batch-data/omicslm_maxtext \\"
echo "      load_parameters_path=gs://omicslm-batch-data/maxtext_checkpoints/qwen3-4b/0/items \\"
echo "      grain_train_files='gs://omicslm-batch-data/arrayrecord/omicslm-*.array_record' \\"
echo "      tokenizer_path=src/maxtext/assets/tokenizers/qwen3-tokenizer \\"
echo "      omics_norm_stats_path=gs://omicslm-batch-data/omics_norm_stats.npz \\"
echo "      steps=5000 per_device_batch_size=4 max_target_length=256 \\"
echo "      enable_checkpointing=true checkpoint_period=1000 \\"
echo "      attention=flash scan_layers=True remat_policy=full \\"
echo "      gradient_clipping_threshold=1.0 \\"
echo "      enable_tensorboard=true log_period=10"
