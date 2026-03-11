#!/usr/bin/env bash
set -euo pipefail

# 改成当前环境的本地 CATLASS 绝对路径
CATLASS_DIR="/path/to/your/catlass"

# 模型运行依赖框架脚本/benchmark/run.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}/../../"
echo $PWD

(
  # aclgraph优化开关 仅在子 shell 生效，结束自动失效
  export ENABLE_ACLGRAPH=1
  export TRITON_INDEX_FUSION=1
  export TRITON_EMBEDDING_FUSION=1
  export CATLASS_EVG_FUSION=1
  export TORCHINDUCTOR_MAX_AUTOTUNE=1
  export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="CATLASS,ATen"
  export TORCHINDUCTOR_NPU_CATLASS_DIR="${CATLASS_DIR}"
  export INDUCTOR_NPU_CATLASS_BENCH_USE_PROFILING=1
  export CATLASS_EPILOGUE_FUSION=1
  export TORCHINDUCTOR_CATLASS_ENABLED_OPS="mm,addmm,bmm,grouped_mm"

  export INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE=1
  export TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING=1
  export INDUCTOR_TINY_KERNEL=1
  export TORCHNPU_PRECOMPILE_THREADS=60

  echo "开启优化开关 ENABLE_ACLGRAPH=${ENABLE_ACLGRAPH}"

  python run.py SIM.json
)
echo "运行结束，优化开关已关闭 ENABLE_ACLGRAPH=${ENABLE_ACLGRAPH:-0}"
