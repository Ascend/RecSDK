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
set(CMAKE_CXX_STANDARD 14)
project(attention)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
include_directories(/usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow/include)
link_directories(/usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow/)

file(GLOB_RECURSE TF_OPS ./*.cpp)
add_library(attention_ops SHARED ${TF_OPS})


target_link_libraries(attention_ops /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow/libtensorflow_framework.so.2)
target_include_directories(attention_ops PUBLIC /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow/include)


install(TARGETS attention_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})