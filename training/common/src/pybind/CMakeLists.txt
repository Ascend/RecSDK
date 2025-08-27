# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
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

cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)

pybind11_add_module(common_binding common_binding.cpp)
set_target_properties(common_binding PROPERTIES LINK_FLAGS "-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack")
target_include_directories(common_binding
        PUBLIC
        ${ASCEND_DRIVER_PATH}/include
        ${ASCEND_DRIVER_PATH}/kernel/inc)
target_link_directories(common_binding
        PUBLIC
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${ASCEND_PATH}/fwkacllib/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/runtime/lib64)
target_link_libraries(common_binding PUBLIC ascendcl ge_executor gert runtime ge_common acl_tdt_channel acl_tdt_queue drvdsmi_host)
install(TARGETS common_binding LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
