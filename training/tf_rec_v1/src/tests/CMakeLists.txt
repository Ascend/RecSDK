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

# 开启测试
enable_testing()
set(CMAKE_CXX_STANDARD 17)
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
add_definitions(-DGTEST)

set(EMOCK_INCLUDE_DIRS /usr/local/include/emock)
include_directories(${EMOCK_INCLUDE_DIRS})
find_library(EMOCK_LIBRARY emock)
if(NOT EMOCK_LIBRARY)
    set(EMOCK_LIBRARY /usr/local/lib/libemock.a)
endif()
message("EMOCK_LIBRARY: " ${EMOCK_LIBRARY})
# src
file(GLOB_RECURSE MXREC_SRC ${PROJECT_SOURCE_DIR}/core/*.cpp)
file(GLOB_RECURSE LCCL_SRC ${PROJECT_SOURCE_DIR}/lccl/*.cpp)
list(APPEND MXREC_SRC ${LCCL_SRC})
message("MXREC_SRC: " ${MXREC_SRC})

file(GLOB_RECURSE MXREC_TEST_SRC ./*.cpp)
message("MXREC_TEST_SRC: " ${MXREC_TEST_SRC})

set(CMAKE_CXX_FLAGS "--coverage")

link_libraries(stdc++fs)

add_executable(test_main ${MXREC_SRC} ${MXREC_TEST_SRC})

if(NOT SECUREC_PATH)
    set(SECUREC_PATH ${PROJECT_SOURCE_DIR}/../platform/securec)
endif()

include_directories(PRIVATE .)

IF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    SET(ARCH aarch64)
ELSEIF (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
    SET(ARCH x86_64)
ENDIF ()

set(TRAINING_COMM_PATH ${PROJECT_SOURCE_DIR}/../../common)
target_include_directories(test_main
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../core
        ${TRAINING_COMM_PATH}/src/core
        ${TRAINING_COMM_PATH}/src/core/pcie_through
        ${TRAINING_COMM_PATH}/src/core/lccl/include
        ${TRAINING_COMM_PATH}/src/core/lccl/src/tools/socket
        ${TRAINING_COMM_PATH}/src/core/lccl/src/3rdparty
        ${ASCEND_DRIVER_PATH}/include
        ${ASCEND_DRIVER_PATH}/kernel/inc
        ${SECUREC_PATH}/include
        ${ASCEND_PATH}/include/
        ${ASCEND_PATH}/${ARCH}-linux/include/hccl
        ${ASCEND_PATH}/${ARCH}-linux/include/experiment/runtime
        ${ASCEND_PATH}/${ARCH}-linux/include/experiment/msprof
)

target_link_directories(test_main
    PRIVATE
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${SECUREC_PATH}/lib)

target_link_directories(test_main
        PUBLIC
        ${ASCEND_PATH}/fwkacllib/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/runtime/lib64
        ${HDF5_PATH}/lib
        ${SECUREC_PATH}/lib
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${TRAINING_COMM_PATH}/src/build/core
        )

target_link_libraries(test_main PUBLIC ${TF_LIB}
        securec
        OpenMP::OpenMP_CXX ${HDF5_CXX_LIBRARIES}  ${MPI_CXX_LIBRARIES}
        ${PYTHON_LIBRARY} drvdsmi_host
        ${ASCEND_PATH}/lib64
        c_sec
        -ldl
        )

target_link_libraries(test_main PUBLIC
        ${GTEST_BOTH_LIBRARIES}
        MPI::MPI_CXX)

target_link_libraries(test_main PUBLIC ascendcl msprofiler ge_executor gert runtime ge_common register graph ascend_protobuf
        profapi opt_feature error_manager exe_graph acl_tdt_channel acl_tdt_queue securec drvdsmi_host ${EMOCK_LIBRARY} gmock libascend_hal.so
        COMMON_ASC)
