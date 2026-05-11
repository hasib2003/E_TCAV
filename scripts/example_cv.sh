#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Usage:
#
# bash  example_cv.sh \
#     <dataset> \
#     <model> \
#     <classifier>
#
# Example:
#
# bash  example_cv.sh scdb resnet50 default
# bash  example_cv.sh celeba inception_v3 signal
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
INPUT_TYPE="image"

# ============================================================
# Dataset Configs
# ============================================================

declare -A DATASET_CONFIGS
declare -A NUM_CLASSES

DATASET_CONFIGS["celeba"]="${SRC_DIR}/config/celeba.json"
DATASET_CONFIGS["isic"]="${SRC_DIR}/config/isic.json"
DATASET_CONFIGS["scdb"]="${SRC_DIR}/config/scdb/config.json"

NUM_CLASSES["celeba"]="2"
NUM_CLASSES["isic"]="1"
NUM_CLASSES["scdb"]="2"

# ============================================================
# Layer Definitions
# ============================================================

declare -A MODEL_LAYERS

MODEL_LAYERS["resnet50"]="layer2.3 layer3.0 layer3.1 layer3.2 layer3.3 layer3.4 layer3.5 layer4.0 layer4.1 layer4.2 avgpool"

MODEL_LAYERS["resnet18"]="layer3.0 layer3.1 layer4.0 layer4.1 avgpool"

MODEL_LAYERS["densenet121"]="features.denseblock4.denselayer5.conv2 \
features.denseblock4.denselayer6.conv2 \
features.denseblock4.denselayer7.conv2 \
features.denseblock4.denselayer8.conv2 \
features.denseblock4.denselayer9.conv2 \
features.denseblock4.denselayer10.conv2 \
features.denseblock4.denselayer11.conv2 \
features.denseblock4.denselayer13.conv2 \
features.denseblock4.denselayer14.conv2 \
features.denseblock4.denselayer15.conv2 \
features.denseblock4.denselayer16.conv2 \
avgpool"

MODEL_LAYERS["inception_v3"]="Mixed_5b Mixed_5c Mixed_5d \
Mixed_6a Mixed_6b Mixed_6c Mixed_6d Mixed_6e \
Mixed_7a Mixed_7b Mixed_7c avgpool"

# ============================================================
# Checkpoints
# ============================================================

declare -A CHECKPOINTS

CHECKPOINTS["resnet50-celeba"]="/path/to/celeba/resnet50/best_model.pt"
CHECKPOINTS["inception_v3-celeba"]="/path/to/celeba/inception_v3/best_model.pt"

CHECKPOINTS["resnet50-isic"]="/path/to/isic/resnet50/best_model.pt"
CHECKPOINTS["inception_v3-isic"]="/path/to/isic/inception_v3/best_model.pt"

CHECKPOINTS["resnet50-scdb"]="/path/to/scdb/resnet50/best.pth"
CHECKPOINTS["inception_v3-scdb"]="/path/to/scdb/inception_v3/best.pth"

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
    --layers ${LAYERS} \
    --checkpoint "${CHECKPOINT}" \
    --num_classes "${NUM_CLASS}" \
    --save_dir "${SAVE_DIR}"