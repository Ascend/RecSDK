# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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
cmake_minimum_required(VERSION 3.5.1)
project(broadcast_equal_pass)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

set(common_compile_options
    -std=c++11
    -Wall
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0
)

############ libfuse_tile_concat_pass.so ############
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

set_target_properties(${PROJECT_NAME} PROPERTIES output_name ${PROJECT_NAME})

target_compile_options(${PROJECT_NAME} PRIVATE
    ${common_compile_options}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${common_compile_definitions}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
    ${ASCEND_PATH}/include/graph
    ${ASCEND_PATH}/include/ge
    ${ASCEND_PATH}/include/register
    ${ASCEND_PATH}/include
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/compiler/lib64
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    -Wl,--no-as-needed
    graph
    register
    -Wl,--as-needed
)
