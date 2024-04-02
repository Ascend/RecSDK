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
project(MxRec LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

if (DEFINED OMPI_PATH)
    message("== ompi path: " ${OMPI_PATH})
else ()
    message("ERROR no OMPI_PATH")
endif ()
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

set(CMAKE_PREFIX_PATH ${OMPI_PATH} ${HDF5_PATH})
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    find_package(easy_profiler)
else ()
    find_package(easy_profiler PATHS ${EASY_PROFILER_PATH} NO_DEFAULT_PATH)
endif ()
if (NOT easy_profiler_FOUND)
    message("===EASY_PROFILER_NOT_FOUND===")
else ()
    message("==EASY_PROFILER_FOUND===")
    ADD_DEFINITIONS(-DBUILD_WITH_EASY_PROFILER)
endif ()
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -ffunction-sections -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -ffunction-sections -O3 -Wfatal-errors -DNDEBUG -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -s")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack")

option(ENABLE_DEBUG "use debug mode" OFF)
if (ENABLE_DEBUG)
    message(ENABLE_DEBUG)
    add_definitions(-DENABLE_DEBUG)
endif ()

message("== ASCEND_PATH: ${ASCEND_PATH}")


include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${ASCEND_PATH}/fwkacllib/include)
include_directories(${ASCEND_PATH}/runtime/include)
include_directories(${ASCEND_PATH}/compiler/include)
include_directories(${PROJECT_SOURCE_DIR}/core)

if (DEFINED TF_PATH)
    message("== TF_PATH: ${TF_PATH}")
    include_directories(${TF_PATH}/include)
    link_directories(${TF_PATH})
else ()
    message("ERROR no TF_PATH try `find / -name libtensorflow_framework.so.1`")
endif ()

if (DEFINED ASCEND_DRIVER_PYTH)
    message(ASCEND_DRIVER_PATH ${ASCEND_DRIVER_PYTH})
else ()
    set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
endif ()

if (${TF_PATH} MATCHES "tensorflow_core")
    message("== Enable tf1 ==")
    link_directories(PUBLIC ${ASCEND_PATH}/../../tfplugin/latest/python/site-packages/npu_bridge/)
    link_directories(PUBLIC ${PYTHON_PATH}/lib/python3.7/site-packages/npu_bridge/)
    set(TF_LIB "-l:libtensorflow_framework.so.1")
else ()
    message("== Enable tf2 ==")
    link_directories(PUBLIC ${ASCEND_PATH}/../../tfplugin/latest/python/site-packages/npu_device/compat/v1/)
    link_directories(PUBLIC ${PYTHON_PATH}/lib/python3.7/site-packages/npu_device/compat/v1/)
    set(TF_LIB "-l:libtensorflow_framework.so.2")
endif ()

if (OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
if(NOT OPENSOURCE_DIR)
    set(OPENSOURCE_DIR ${PROJECT_SOURCE_DIR}/../../opensource/)
endif()

if(IS_DIRECTORY ${OPENSOURCE_DIR})
    add_subdirectory(${OPENSOURCE_DIR}/pybind11 pybind11.out)
else()
    message(FATAL_ERROR "INVALID FOLDER, ${OPENSOURCE_DIR}")
endif()

add_subdirectory(core)
add_subdirectory(ops_tf)
add_subdirectory(pybind)
add_subdirectory(dataset_tf)
if (CMAKE_BUILD_TYPE MATCHES "Release")
    message(STATUS "CMAKE_BUILD_TYPE is Release")
else()
    message(STATUS "CMAKE_BUILD_TYPE is Debug. [default]")
    add_subdirectory(tests)
endif ()
