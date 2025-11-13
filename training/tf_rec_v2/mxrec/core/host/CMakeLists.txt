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

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(TRAINING_COMM_PATH ${PROJECT_SOURCE_DIR}/../common)
include_directories(${TF_DIR}/include)
include_directories(${ASCEND_DIR}/include)
include_directories(${PYBIND11_DIR}/include)
include_directories(${SPDLOG_DIR}/include)
include_directories(${TRAINING_COMM_PATH}/src/core)

file(GLOB_RECURSE SRC_FILES *.cpp)
list(REMOVE_ITEM SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/binding.cpp)

add_library(host STATIC ${SRC_FILES})
target_include_directories(host
    PUBLIC
        ${ASCEND_HOME_PATH}/include/hccl
        ${ASCEND_HOME_PATH}/include/experiment/msprof/
        ${ASCEND_HOME_PATH}/include/experiment/runtime/
        ${CMAKE_CURRENT_SOURCE_DIR}/lccl/src/3rdparty/
        ${CMAKE_CURRENT_SOURCE_DIR}/lccl/include
)
target_link_directories(host PUBLIC ${TRAINING_COMM_PATH}/src/build/core)
target_link_libraries(host PRIVATE ${TF_LIB})
target_link_libraries(host PRIVATE ${ACL_LIB})
target_link_libraries(host PRIVATE ${ACL_TDT_LIB})
target_link_libraries(host PRIVATE spdlog)
target_link_libraries(host PRIVATE stdc++fs)
target_link_libraries(host PRIVATE COMMON_ASC)

pybind11_add_module(binding binding.cpp)
target_link_libraries(binding PRIVATE host)
