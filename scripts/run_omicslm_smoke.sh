#!/usr/bin/env bash
# Copyright 2024–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Smoke / integration run for OmicsLM instruction tuning.
#
# Runs run_omicslm_train.py with synthetic data — no external datasets,
# checkpoints, or cloud credentials required.  Works on a single GPU, a
# multi-GPU host, or a TPU VM (JAX picks up devices automatically).
#
# Optional env overrides (all have safe defaults):
#   VENV_PATH       path to the virtualenv to activate (default: ./maxtext_venv)
#   STEPS           training steps          (default: 50)
#   BATCH_SIZE      batch size              (default: 2)
#   SEQ_LEN         sequence length         (default: 64)
#   LEARNING_RATE   learning rate           (default: 1e-4)
#   HIDDEN_SIZE     LLM hidden dimension    (default: 64  — use 4096 for 7B)
#   NUM_LAYERS      transformer layers      (default: 2)
#   OMICS_DIM       omics input dimension   (default: 21287)
#   OUTPUT_DIR      checkpoint output path  (default: /tmp/omicslm_out)
#
# Usage:
#   bash scripts/run_omicslm_smoke.sh
#
#   # Override any variable inline:
#   STEPS=200 HIDDEN_SIZE=256 bash scripts/run_omicslm_smoke.sh

set -euo pipefail

MAXTEXT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

VENV_PATH="${VENV_PATH:-${MAXTEXT_ROOT}/maxtext_venv}"
STEPS="${STEPS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
OMICS_DIM="${OMICS_DIM:-21287}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/omicslm_out}"

# Activate virtualenv if it exists; otherwise assume the environment is already set up.
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090,SC1091
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${MAXTEXT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== OmicsLM smoke run ==="
echo "  steps         : ${STEPS}"
echo "  batch_size    : ${BATCH_SIZE}"
echo "  seq_len       : ${SEQ_LEN}"
echo "  learning_rate : ${LEARNING_RATE}"
echo "  hidden_size   : ${HIDDEN_SIZE}"
echo "  num_layers    : ${NUM_LAYERS}"
echo "  omics_dim     : ${OMICS_DIM}"
echo "  output_dir    : ${OUTPUT_DIR}"
echo "  JAX devices   : $(python -c 'import jax; print(jax.devices())')"
echo

python "${MAXTEXT_ROOT}/scripts/run_omicslm_train.py" \
  --steps "${STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --seq_len "${SEQ_LEN}" \
  --learning_rate "${LEARNING_RATE}" \
  --hidden_size "${HIDDEN_SIZE}" \
  --num_layers "${NUM_LAYERS}" \
  --omics_dim "${OMICS_DIM}" \
  --output_dir "${OUTPUT_DIR}"

echo
echo "=== OmicsLM smoke run PASSED ==="
