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

# TorchRec 测试覆盖率脚本
# 功能：运行所有测试并生成包含测试结果的HTML报告

echo "========================================="
echo "TorchRec 测试覆盖率统计"
echo "========================================="

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 创建覆盖率配置文件（如果文件不存在）
echo -e "${YELLOW}检查覆盖率配置文件...${NC}"
if [ ! -f .coveragerc ]; then
    echo -e "${YELLOW}配置文件不存在，正在创建...${NC}"
    cat > .coveragerc << 'EOF'
[run]
# 指定要测量的源代码
source = torchrec
# 并行测量（支持多进程测试）
parallel = True
# 测量分支覆盖率
branch = True
# 排除测试文件、示例和配置文件
omit =
    */test_*
    */tests/*
    */testing/*
    *_test.py
    tests/*
    examples/*
    docs/*
    setup.py
    conftest.py

[report]
# 显示缺失覆盖率的行号
show_missing = True
# 在报告中忽略匹配的文件
omit =
    */test_*
    */tests/*
    */testing/*
    *_test.py
    tests/*
    examples/*
    docs/*
    setup.py
    conftest.py
# 输出格式
precision = 2

[html]
# HTML 报告目录
directory = coverage_html_report

[xml]
# XML 报告文件
output = coverage.xml
EOF
    echo -e "${GREEN}配置文件创建完成${NC}"
else
    echo -e "${GREEN}配置文件已存在，跳过创建${NC}"
fi

# 清理之前的覆盖率数据
echo -e "${YELLOW}清理之前的覆盖率数据...${NC}"
coverage erase
rm -rf .coverage.* coverage_html_report
echo -e "${GREEN}清理完成${NC}"

# 运行测试并收集覆盖率
echo -e "${YELLOW}开始运行测试并收集覆盖率...${NC}"
echo "========================================="

# 创建带时间戳的测试报告目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_REPORT_DIR="test_reports_${TIMESTAMP}"
mkdir -p ${TEST_REPORT_DIR}

python3 -m pytest \
    ./torchrec/sparse/tests/*.py \
    ./torchrec/modules/tests/test_embedding_modules.py \
    ./torchrec/distributed/planner/tests/*.py \
    ./torchrec/inference/tests/test_inference.py \
    --cov=torchrec \
    --cov-report=html:${TEST_REPORT_DIR}/coverage_html_report \
    --cov-report=xml:${TEST_REPORT_DIR}/coverage.xml \
    --cov-config=.coveragerc \
    --html=${TEST_REPORT_DIR}/test_results.html \
    --self-contained-html \
    -v --tb=short

# 保存退出状态
EXIT_CODE=$?

echo "========================================="

# 检查测试是否成功
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}所有测试通过！${NC}"
else
    echo -e "${RED}部分测试失败，请检查上面的输出${NC}"
fi

# 合并覆盖率数据（如果使用并行测试）
if ls .coverage.* 1> /dev/null 2>&1; then
    echo -e "${YELLOW}合并并行覆盖率数据...${NC}"
    coverage combine
fi

exit $EXIT_CODE
