#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

platform="${1:-}"
precision="${2:-bf16}"
if [[ "$platform" != "gpu" && "$platform" != "npu" ]]; then
    echo "usage: bash run_v2_full_training.sh <gpu|npu> [fp32|bf16]" >&2
    exit 2
fi
if [[ "$precision" != "fp32" && "$precision" != "bf16" ]]; then
    echo "usage: bash run_v2_full_training.sh <gpu|npu> [fp32|bf16]" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-python}"
events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
registry_path="${SID_REGISTRY_PATH:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint_path="${V2_FULL_TRAIN_CHECKPOINT:-checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt}"
output_root="${V2_FULL_TRAIN_OUTPUT_ROOT:-checkpoints/full_training/v2_full}"
epochs="${V2_FULL_TRAIN_EPOCHS:-1}"
batch_size="${V2_FULL_TRAIN_BATCH_SIZE:-64}"

if [[ "$platform" == "gpu" ]]; then
    device="${CUDA_DEVICE:-cuda:0}"
else
    device="${NPU_DEVICE:-npu:0}"
fi

for required in "$events_path" "$registry_path" "$checkpoint_path"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR missing required input: $required" >&2
        exit 3
    fi
done

output_dir="$output_root/$platform/$precision"
mkdir -p "$output_dir"
extra_args=()
if [[ -n "${V2_FULL_TRAIN_MAX_SAMPLES:-}" ]]; then
    extra_args+=(--max-train-samples "$V2_FULL_TRAIN_MAX_SAMPLES")
fi
if [[ -n "${V2_FULL_TRAIN_LEARNING_RATE:-}" ]]; then
    extra_args+=(--learning-rate "$V2_FULL_TRAIN_LEARNING_RATE")
fi

"$python_bin" scripts/train_v2_full_epochs.py \
    --platform "$platform" \
    --device "$device" \
    --precision "$precision" \
    --events "$events_path" \
    --checkpoint "$checkpoint_path" \
    --sid-registry "$registry_path" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --output-dir "$output_dir" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$output_dir/full_training.log"
