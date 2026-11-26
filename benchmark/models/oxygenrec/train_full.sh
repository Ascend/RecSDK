#!/usr/bin/env bash
set -euo pipefail

# The benchmark starts this wrapper in the pinned OxygenREC checkout.
python_bin="${PYTHON_BIN:-python3}"
assets="${OXYGENREC_ASSET_ROOT:-$(cd ../../.. && pwd)/datasets/oxygenrec}"
raw="$assets/data/raw/retailrocket"
events="${EVENTS_PATH:-$raw/events.csv}"
registry="${SID_REGISTRY_PATH:-$assets/data/processed/rq_comparison/w256_kmeanspp/sid_registry.json}"
checkpoint_override="${V2_FULL_TRAIN_CHECKPOINT:-${V2_FULL_CHECKPOINT:-}}"
checkpoint="${checkpoint_override:-$assets/checkpoints/gpu_pretraining/full-epoch-1.pt}"

# Let the existing launcher handle unsupported training options before preparing data.
if [[ "${MODEL_CPU_ONLY:-False}" == True || "${MODEL_COMPILE_FLAG:-False}" == True ]]; then
    exec bash run_oxygenrec.sh train
fi
case "${OXYGENREC_PRECISION:-${MODEL_DATA_TYPE:-bf16}}" in
    bf16|bfloat16|fp32|float32) ;;
    *) exec bash run_oxygenrec.sh train ;;
esac

prepare_assets() (
    echo "Preparing OxygenREC assets under $assets"
    lock="$assets/.prepare.lock"
    sid_stage=""
    checkpoint_stage=""
    mkdir -p -- "$assets"
    if ! mkdir -- "$lock"; then
        echo "ERROR unable to lock OxygenREC assets: $lock" >&2
        exit 3
    fi
    cleanup() {
        if [[ -n "$sid_stage" ]]; then
            rm -r -- "$sid_stage"
        fi
        if [[ -n "$checkpoint_stage" ]]; then
            rm -r -- "$checkpoint_stage"
        fi
        rmdir -- "$lock"
    }
    trap cleanup EXIT

    prep="../original/prepare"
    if [[ ! -f "$registry" ]]; then
        for property in "$raw/item_properties_part1.csv" "$raw/item_properties_part2.csv"; do
            if [[ ! -f "$property" ]]; then
                echo "ERROR missing RetailRocket property file: $property" >&2
                exit 3
            fi
        done
        mkdir -p -- "$(dirname "$registry")"
        sid_stage="$(mktemp -d "$(dirname "$registry")/.oxygenrec-sid-XXXXXXXX")"
        "$python_bin" -u "$prep/fit_retailrocket_sid.py" --device cpu --events "$events" \
            --properties "$raw/item_properties_part1.csv" "$raw/item_properties_part2.csv" \
            --width 256 --initialization kmeans++ --output-dir "$sid_stage"
        mv -T -- "$sid_stage/sid_registry.json" "$registry"
    fi

    if [[ ! -f "$checkpoint" ]]; then
        mkdir -p -- "$(dirname "$checkpoint")"
        checkpoint_stage="$(mktemp -d "$(dirname "$checkpoint")/.oxygenrec-checkpoint-XXXXXXXX")"
        "$python_bin" -u "$prep/train_v2_pretraining_ablation_retailrocket.py" \
            --device cpu --events "$events" --sid-registry "$registry" --epochs 1 \
            --max-train-samples 5000 --max-validation-samples 32 --output-dir "$checkpoint_stage"
        mv -T -- "$checkpoint_stage/full-epoch-1.pt" "$checkpoint"
    fi
)

# Keep validation of caller-supplied inputs in the Full launcher.
if [[ ! -f "$events" || ( -f "$registry" && -f "$checkpoint" ) ]]; then
    exec bash run_oxygenrec.sh train
fi
if [[ ! -f "$checkpoint" && -n "$checkpoint_override" ]]; then
    exec bash run_oxygenrec.sh train
fi
if [[ ! -f "$registry" && ( -n "${SID_REGISTRY_PATH:-}" || -f "$checkpoint" ) ]]; then
    exec bash run_oxygenrec.sh train
fi
prepare_assets

exec bash run_oxygenrec.sh train
