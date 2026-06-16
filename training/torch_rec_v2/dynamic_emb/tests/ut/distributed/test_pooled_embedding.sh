#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
DIST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYHKV_DEBUG=1
export PYHKV_DEBUG_ITER=10

torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  "${SCRIPT_DIR}/test_pooled_embedding_fw.py" --print_sharding_plan --optimizer_type "adam"
