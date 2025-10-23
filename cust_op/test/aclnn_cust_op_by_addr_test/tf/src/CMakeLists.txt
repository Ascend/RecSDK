# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_execute)

# Compile options
add_compile_options(-std=c++14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../output")

set(INC_PATH $ENV{DDK_PATH})
set(AUTO_GEN_PATH "../../../../ascendc_op/ai_core_op/cust_op_by_addr/v220/cust_op_by_addr/build_out/autogen")

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "${INC_PATH}/acllib/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ${INC_PATH}/runtime/include
    ${INC_PATH}/atc/include
        ../inc
    ${AUTO_GEN_PATH}
)

# add host lib path
link_directories(
    ${LIB_PATH}
)

add_executable(execute_op
    ${AUTO_GEN_PATH}/aclnn_embedding_lookup_by_address.cpp
    ${AUTO_GEN_PATH}/aclnn_embedding_update_by_address.cpp
    operator_desc.cpp
    op_runner.cpp
    op_runner_lookup.cpp
    op_runner_update.cpp
    main.cpp
    common.cpp
)

target_link_libraries(execute_op
    ascendcl
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS execute_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
