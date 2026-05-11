#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Usage
#
# bash  example_text.sh \
#     <dataset> \
#     <model> \
#     <classifier>
#
# Example:
#
# bash  example_text.sh wiki roberta-base default
# bash  example_text.sh wiki roberta-base signal
#
# ============================================================

DATASET="${1:?Missing dataset}"
MODEL="${2:?Missing model}"
CLASSIFIER="${3:-default}"

# ============================================================
# Paths
# ============================================================

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="${BASE_DIR}/src"

OUTPUT_ROOT="${BASE_DIR}/outputs"

# ============================================================
# Static Config
# ============================================================

TCAV_MODE="default"
INPUT_TYPE="text"
MAX_LEN=256

# ============================================================
# Dataset Configs
# ============================================================

declare -A DATASET_CONFIGS
declare -A NUM_CLASSES

DATASET_CONFIGS["wiki"]="${SRC_DIR}/config/production-1/extended/wiki-extended.json"

NUM_CLASSES["wiki"]="2"

# ============================================================
# Layer Definitions
# ============================================================

declare -A MODEL_LAYERS

MODEL_LAYERS["roberta-base"]="\
roberta.encoder.layer.4.output \
roberta.encoder.layer.5.output \
roberta.encoder.layer.6.output \
roberta.encoder.layer.7.output \
roberta.encoder.layer.8.output \
roberta.encoder.layer.9.output \
roberta.encoder.layer.10.output \
roberta.encoder.layer.11.output \
classifier.dense \
classifier.dropout"

# ============================================================
# Checkpoints
# ============================================================

declare -A CHECKPOINTS

CHECKPOINTS["roberta-base-wiki"]="/path/to/checkpoints/wiki/best_model.pth"

# ============================================================
# Validation
# ============================================================

if [[ -z "${DATASET_CONFIGS[$DATASET]:-}" ]]; then
    echo "Unknown dataset: ${DATASET}"
    exit 1
fi

if [[ -z "${MODEL_LAYERS[$MODEL]:-}" ]]; then
    echo "Unknown model: ${MODEL}"
    exit 1
fi

CHECKPOINT_KEY="${MODEL}-${DATASET}"

if [[ -z "${CHECKPOINTS[$CHECKPOINT_KEY]:-}" ]]; then
    echo "Missing checkpoint for ${CHECKPOINT_KEY}"
    exit 1
fi

# ============================================================
# Derived Config
# ============================================================

CONCEPT_CONFIG="${DATASET_CONFIGS[$DATASET]}"
LAYERS="${MODEL_LAYERS[$MODEL]}"
CHECKPOINT="${CHECKPOINTS[$CHECKPOINT_KEY]}"
NUM_CLASS="${NUM_CLASSES[$DATASET]}"

SAVE_DIR="${OUTPUT_ROOT}/${CLASSIFIER}/${DATASET}/${MODEL}"

mkdir -p "${SAVE_DIR}"

# ============================================================
# Run
# ============================================================

echo "=========================================="
echo "Dataset     : ${DATASET}"
echo "Model       : ${MODEL}"
echo "Classifier  : ${CLASSIFIER}"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Save Dir    : ${SAVE_DIR}"
echo "=========================================="

cd "${SRC_DIR}"

python -m pipelines.main \
    --tcav_mode "${TCAV_MODE}" \
    --classifier "${CLASSIFIER}" \
    --input_type "${INPUT_TYPE}" \
    --concept_config "${CONCEPT_CONFIG}" \
    --model "${MODEL}" \
    --tokenizer "${MODEL}" \
    --max_len "${MAX_LEN}" \
    --layers ${LAYERS} \
    --checkpoint "${CHECKPOINT}" \
    --num_classes "${NUM_CLASS}" \
    --save_dir "${SAVE_DIR}"