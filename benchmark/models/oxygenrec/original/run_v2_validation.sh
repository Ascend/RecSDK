#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

platform="${1:-}"
precision="${2:-all}"
if [[ "$platform" != "gpu" && "$platform" != "npu" ]]; then
    echo "usage: bash run_v2_validation.sh <gpu|npu> [fp32|bf16|all]" >&2
    exit 2
fi
if [[ "$precision" != "fp32" && "$precision" != "bf16" && "$precision" != "all" ]]; then
    echo "usage: bash run_v2_validation.sh <gpu|npu> [fp32|bf16|all]" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-python}"
events_path="${EVENTS_PATH:-data/raw/retailrocket/events.csv}"
registry_path="${SID_REGISTRY_PATH:-data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint_path="${V2_FULL_CHECKPOINT:-checkpoints/retailrocket_v2_pretraining_ablation_smoke/full-epoch-1.pt}"
output_root="${V2_VALIDATION_ROOT:-checkpoints/device_validation/v2_full}"
samples="${V2_VALIDATION_SAMPLES:-32}"
batch_size="${V2_VALIDATION_BATCH_SIZE:-32}"
beam_width="${V2_VALIDATION_BEAM_WIDTH:-5}"

if [[ "$platform" == "gpu" ]]; then
    device="${CUDA_DEVICE:-cuda:0}"
else
    device="${NPU_DEVICE:-npu:0}"
fi

missing=0
for required in "$events_path" "$registry_path" "$checkpoint_path"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR missing required input: $required" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    exit 3
fi

if [[ "$precision" == "all" ]]; then
    precisions=(fp32 bf16)
else
    precisions=("$precision")
fi

for current_precision in "${precisions[@]}"; do
    output_dir="$output_root/$platform/$current_precision"
    mkdir -p "$output_dir"
    cases_args=()
    if [[ "${V2_VALIDATION_CASES:-0}" == "1" ]]; then
        cases_args+=(--cases-output "$output_dir/v2_full_${platform}_${current_precision}_validation_cases.json")
    fi
    "$python_bin" scripts/evaluate_v2_fixed_validation.py \
        --platform "$platform" \
        --device "$device" \
        --precision "$current_precision" \
        --events "$events_path" \
        --checkpoint "$checkpoint_path" \
        --sid-registry "$registry_path" \
        --samples "$samples" \
        --batch-size "$batch_size" \
        --beam-width "$beam_width" \
        "${cases_args[@]}" \
        --output "$output_dir/v2_full_${platform}_${current_precision}_validation_summary.json" \
        2>&1 | tee "$output_dir/validation.log"
done
