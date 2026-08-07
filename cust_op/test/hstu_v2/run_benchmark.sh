#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Backward benchmark
msprof --app="python3 test_backward_op_benchmark.py" --output=./prof

python3 utils/perf_stats.py --input-dir ./prof --target-ops HstuBackwardV2 --output ./benchmark_result.csv

rm -rf ./prof
rm -f tmp_benchmark.csv

# Forward benchmark
msprof --app="python3 test_forward_op_benchmark.py" --output=./prof

python3 utils/perf_stats.py --input-dir ./prof --target-ops HstuForwardV2 --output ./benchmark_result_forward.csv

rm -rf ./prof
rm -f tmp_benchmark.csv
