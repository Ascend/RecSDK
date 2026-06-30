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

cmake_minimum_required(VERSION 3.14)
project(test_hstu_dense_forward LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pthread for std::thread
find_package(Threads REQUIRED)

# ASCEND toolchain
set(CANN "$ENV{ASCEND_TOOLKIT_HOME}" CACHE PATH "CANN install path")
set(OPV  "${CANN}/opp/vendors/hstu_dense_forward")

include_directories(
    "${CANN}/include"
    "${OPV}/op_api/include"
)

link_directories(
    "${CANN}/lib64"
    "${OPV}/op_api/lib"
)

add_executable(test_hstu_dense_forward test_hstu_dense_forward.cpp)

target_link_libraries(test_hstu_dense_forward PRIVATE
    ascendcl
    hccl
    c_sec
    cust_opapi
    nnopbase
    Threads::Threads
)

set_target_properties(test_hstu_dense_forward PROPERTIES
    BUILD_RPATH "${CANN}/lib64:${OPV}/op_api/lib"
    INSTALL_RPATH "${CANN}/lib64:${OPV}/op_api/lib"
)

install(TARGETS test_hstu_dense_forward
    RUNTIME DESTINATION bin
)
