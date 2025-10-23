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
cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 17)

pybind11_add_module(custom_pybind module_main.cpp)
set_target_properties(custom_pybind PROPERTIES LINK_FLAGS "-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack")
target_include_directories(custom_pybind PUBLIC ${ASCEND_DRIVER_PATH}/include)
target_link_directories(custom_pybind PUBLIC ${ASCEND_DRIVER_PATH}/lib64/driver)
target_link_libraries(custom_pybind PUBLIC SRC)
install(TARGETS custom_pybind LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})