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

cmake_minimum_required(VERSION 3.20)

project(mxrec LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_compile_options(-fPIC)
add_compile_options(-Werror)
add_compile_options(-Wfatal-errors)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)

add_link_options(-pie)
add_link_options(-Wl,-z,noexecstack)
add_link_options(-Wl,-z,relro)
add_link_options(-Wl,-z,now)
add_link_options(-Wl,--build-id=none)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(
    -O3
    -DNDEBUG
    -fstack-protector-all
    -s
  )
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(
    -O0
    -D_DEBUG
    -g
  )
endif()

add_subdirectory(third_party)
add_subdirectory(mxrec)
