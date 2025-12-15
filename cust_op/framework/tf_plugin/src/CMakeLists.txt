# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

file(GLOB_RECURSE OP_TF_PLUGIN_SRC ./*.cpp)
add_library(recsdk_tf_npu_ops SHARED ${OP_TF_PLUGIN_SRC})

target_link_directories(recsdk_tf_npu_ops
    PUBLIC
        ${ASCEND_PATH}/fwkacllib/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/runtime/lib64
        ${SECUREC_PATH}/lib
        ${ASCEND_DRIVER_PATH}/lib64/driver
)

target_link_libraries(recsdk_tf_npu_ops PUBLIC ascendcl msprofiler ge_executor gert runtime ge_common register graph ascend_protobuf
    profapi error_manager exe_graph acl_tdt_channel acl_tdt_queue securec drvdsmi_host)

target_link_libraries(recsdk_tf_npu_ops PUBLIC
        -l:_tf_adapter.so
        OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES}
        ${PYTHON_LIBRARY}
)

install(TARGETS recsdk_tf_npu_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})