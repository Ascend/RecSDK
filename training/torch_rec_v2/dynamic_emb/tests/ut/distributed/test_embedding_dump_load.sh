#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  -m pytest -s \
  "${SCRIPT_DIR}/test_dynamicemb_table_dump_load.py"
