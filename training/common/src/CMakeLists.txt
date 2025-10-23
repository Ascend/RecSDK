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
project(MxRec LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

if (DEFINED CMAKE_INSTALL_PREFIX)
    message("== install in " ${CMAKE_INSTALL_PREFIX})
else ()
    message("ERROR no CMAKE_INSTALL_PREFIX")
endif ()
if (DEFINED PYTHON_PATH)
    set(PYTHON_INCLUDE_DIR ${PYTHON_PATH}/include/python3.7m)
    set(PYTHON_LIBRARY ${PYTHON_PATH}/lib/libpython3.7m.so)
else ()
    message("ERROR no PYTHON_PATH")
endif ()

find_package(PythonLibs 3.7 REQUIRED)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -ffunction-sections -O0 -Wall -g2 -ggdb -fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer -fno-stack-protector")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -ffunction-sections -O3 -Wfatal-errors -DNDEBUG -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -s")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack")

option(ENABLE_DEBUG "use debug mode" OFF)
if (ENABLE_DEBUG)
    message(ENABLE_DEBUG)
    add_definitions(-DENABLE_DEBUG)
endif ()

message("== ASCEND_PATH: ${ASCEND_PATH}")

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${ASCEND_PATH}/fwkacllib/include)
include_directories(${ASCEND_PATH}/runtime/include)
include_directories(${ASCEND_PATH}/compiler/include)
include_directories(${ASCEND_PATH}/include/hccl)
include_directories(${PROJECT_SOURCE_DIR}/core)

if (DEFINED ASCEND_DRIVER_PYTH)
    message(ASCEND_DRIVER_PATH ${ASCEND_DRIVER_PYTH})
else ()
    set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
endif ()

add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
if(NOT OPENSOURCE_DIR)
    set(OPENSOURCE_DIR ${PROJECT_SOURCE_DIR}/../../../../opensource/)
endif()

if(IS_DIRECTORY ${OPENSOURCE_DIR})
    add_subdirectory(${OPENSOURCE_DIR}/pybind11 pybind11.out)
else()
    message(FATAL_ERROR "INVALID FOLDER, ${OPENSOURCE_DIR}")
endif()

add_subdirectory(pybind)
add_subdirectory(core)
if (CMAKE_BUILD_TYPE MATCHES "Release")
    message(STATUS "CMAKE_BUILD_TYPE is Release")
else()
    message(STATUS "CMAKE_BUILD_TYPE is Debug. [default]")
    add_subdirectory(tests)
endif ()
