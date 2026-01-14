# !/bin/bash
# Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
fi

if ! $PYTHON_CMD -c "import tensorflow as tf" &> /dev/null; then
    echo "tensorflow has not been installed"
    exit 1
fi

TF_INCLUDE=$($PYTHON_CMD -c "import tensorflow as tf; print(tf.sysconfig.get_include())")
TF_LIB=$($PYTHON_CMD -c "import tensorflow as tf; print(tf.sysconfig.get_lib())")

if [ ! -d "$TF_INCLUDE" ]; then
    echo "tf include dic does not exist: $TF_INCLUDE"
    exit 1
fi

if [ ! -d "$TF_LIB" ]; then
    echo "tf lib dic does not exist:$TF_LIB"
    exit 1
fi

# g++ 编译命令
g++ -std=c++11 -g -Wall -fPIC -shared \
UserItemFlashAttention.cpp \
-o user_item_flash_attention_op.so \
-D_GLIBCXX_USE_CXX11_ABI=0 \
-I${TF_INCLUDE} \
-I${TF_INCLUDE}/tensorflow/core/platform/linux \
-L${TF_LIB} \
-ltensorflow_framework \
-pthread
