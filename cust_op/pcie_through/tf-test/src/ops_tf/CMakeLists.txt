# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 14)
IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

file(GLOB_RECURSE COMM_OP_SRC ./*.cpp)
add_library(rma_tf_ops SHARED ${COMM_OP_SRC})
target_link_libraries(rma_tf_ops /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/libtensorflow_framework.so.1
        ${ASCEND_PATH}/${ARCH}-linux/lib64/libascendcl.so)
target_include_directories(rma_tf_ops PUBLIC /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/include)
target_include_directories(rma_tf_ops PUBLIC SRC)
install(TARGETS rma_tf_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})