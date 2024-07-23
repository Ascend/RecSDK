# Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cmake_minimum_required(VERSION 3.14.1)

project(ctr CXX C)

if (${BUILD_MODE} MATCHES "release")
    message("======BUILD_MODE release")
    set(CXX_FLAGS
            -O3
            -Wall
            -fPIC
            -fms-extensions
            -Wunused-variable
            -Wunused-value
            -Wcast-align
            -Wcast-qual
            -Winvalid-pch
            -Wwrite-strings
            -Wsign-compare
            -Wfloat-equal
            -Wextra
            -D_FORTIFY_SOURCE=2
            -std=c++17
            -fstack-protector-all
            -fstack-protector-strong
            )
elseif (${BUILD_MODE} MATCHES "debug")
    message("======BUILD_MODE debug")
    set(CXX_FLAGS
            -g
            -O0
            -Wall
            -fPIC
            -fms-extensions
            -Wunused-variable
            -Wunused-value
            -Winvalid-pch
            -Wcast-align
            -Wcast-qual
            -Wwrite-strings
            -Wsign-compare
            -Wfloat-equal
            -Wextra
            -std=c++17
            )
elseif (${BUILD_MODE} MATCHES "ut")
    message("======BUILD_MODE ut")
    set(CXX_FLAGS
            -g
            -Wall
            -fPIC
            -fms-extensions
            -Wunused-variable
            -Wunused-value
            -Winvalid-pch
            -Wcast-align
            -Wcast-qual
            -Wwrite-strings
            -Wsign-compare
            -Wfloat-equal
            -Wextra
            -std=c++17
            -fsanitize=address
            -fsanitize-recover=address,all
            -fno-omit-frame-pointer
            -fstack-protector-all
            )
else ()
    message(FATAL_ERROR "======BUILD_MODE not found")
endif (${BUILD_MODE} MATCHES "release")

string(REPLACE ";" " " CMAKE_CXX_FLAGS "${CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

if (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
    add_definitions(-D__AARCH64__)
    set(CMAKE_ARC linux-aarch64)
elseif (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "x86_64")
    add_definitions(-D__X86_64__)
    set(CMAKE_ARC linux-x86_64)
    set(CXX_FLAGS
            ${CXX_FLAGS}
            -msse2
            -mavx
            )
else ()
    message(FATAL_ERROR "don't support ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif (${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")

set(OCK_CTR_PLATFORM_UTIL_DIR       ${PROJECT_SOURCE_DIR}/../../../opensource)
message(===============${OCK_CTR_PLATFORM_UTIL_DIR})
include_directories(${OCK_CTR_PLATFORM_UTIL_DIR}/securec/include)

include_directories(
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/embedding_cache
)

add_subdirectory(3rdparty)
add_subdirectory(src)

if (${BUILD_MODE} MATCHES "release")
elseif (${BUILD_MODE} MATCHES "debug")
else ()
    add_subdirectory(tests)
endif (${BUILD_MODE} MATCHES "release")