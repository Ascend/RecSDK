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

cmake_minimum_required(VERSION 3.20)

pybind11_add_module(mxrec_pybind module_main.cpp)
set_target_properties(mxrec_pybind PROPERTIES LINK_FLAGS "-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack")
target_include_directories(mxrec_pybind PUBLIC ${ASCEND_DRIVER_PATH}/include)
target_link_directories(mxrec_pybind PUBLIC ${ASCEND_DRIVER_PATH}/lib64/driver)
target_link_libraries(mxrec_pybind PUBLIC ASC lcal)
install(TARGETS mxrec_pybind LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
