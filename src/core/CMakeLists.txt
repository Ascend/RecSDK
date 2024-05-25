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

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)

if(NOT ABSEIL_PATH)
    set(ABSEIL_PATH ${PYTHON_PATH}/lib/python3.7/site-packages/tensorflow_core/)
endif()

message("ABSEIL_PATH: " ${ABSEIL_PATH})

if(NOT SECUREC_PATH)
    set(SECUREC_PATH ${PROJECT_SOURCE_DIR}/../platform/securec)
endif()
message("SECUREC_PATH: " ${SECUREC_PATH})

if(NOT ACCCTR_PATH)
    set(ACCCTR_PATH ${PROJECT_SOURCE_DIR}/AccCTR)
endif()
message("ACCCTR_PATH: " ${ACCCTR_PATH})

include_directories(${ABSEIL_PATH}/include)
link_directories(${ABSEIL_PATH}/lib)

if (${TF_PATH} MATCHES "tensorflow_core")
    include_directories(${PYTHON_PATH}/lib/python3.7/site-packages/tensorflow_core/include)
else()
    include_directories(${PYTHON_PATH}/lib/python3.7/site-packages/tensorflow/include)
endif()

link_libraries(stdc++fs)

file(GLOB_RECURSE MXREC_SRC ./*.cpp ./*.h)
add_library(ASC SHARED ${MXREC_SRC})

target_include_directories(ASC
    PUBLIC
        ${SECUREC_PATH}/include
        ${ASCEND_DRIVER_PATH}/include
)

target_link_directories(ASC
    PUBLIC
        ${ASCEND_PATH}/fwkacllib/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/runtime/lib64
        ${HDF5_PATH}/lib
        ${SECUREC_PATH}/lib
        ${ASCEND_DRIVER_PATH}/lib64/driver
        ${ACCCTR_PATH}/output/ock_ctr_common/lib
)

target_link_libraries(ASC PUBLIC ascendcl msprofiler ge_executor gert runtime ge_common register graph ascend_protobuf
    profapi opt_feature error_manager exe_graph acl_tdt_channel acl_tdt_queue securec drvdsmi_host _ock_ctr_common)
target_link_libraries(ASC PUBLIC
        -l:_tf_adapter.so
        OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES}
        ${PYTHON_LIBRARY}
)
find_package(easy_profiler PATHS ${EASY_PROFILER_PATH} NO_DEFAULT_PATH)
if (easy_profiler_FOUND)
    message("==link with easy_profiler==")
    target_link_directories(ASC PUBLIC ${EASY_PROFILER_PATH})
    target_link_libraries(ASC PUBLIC easy_profiler )
endif ()

install(TARGETS ASC LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
