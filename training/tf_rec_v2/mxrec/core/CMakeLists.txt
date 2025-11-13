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

if(NOT DEFINED PYTHON_HOME)
    message(FATAL_ERROR
        "The environment variable ==> PYTHON_HOME <== is not available, please check your environment settings!"
    )
endif()
set(PYTHON_DIR ${PYTHON_HOME})

if(TF_VERSION STREQUAL "1")
    set(TF_DIR ${PYTHON_DIR}/lib/python3.7/site-packages/tensorflow_core)
    set(TF_LIB ${TF_DIR}/libtensorflow_framework.so.1)
elseif(TF_VERSION STREQUAL "2")
    set(TF_DIR ${PYTHON_DIR}/lib/python3.7/site-packages/tensorflow)
    set(TF_LIB ${TF_DIR}/libtensorflow_framework.so.2)
else()
    message(FATAL_ERROR
        "The environment variable ==> TF_VERSION <== is not set correctly, "
        "TF_VERSION should be set to `1` or `2`!"
    )
endif()

if(NOT DEFINED ASCEND_TOOLKIT_HOME)
    message(FATAL_ERROR
        "The environment variable ==> ASCEND_TOOLKIT_HOME <== is not available, please check your environment settings!"
    )
endif()
set(ASCEND_DIR ${ASCEND_TOOLKIT_HOME})
set(ACL_LIB_PATH ${ASCEND_DIR}/lib64)
set(ACL_LIB ${ASCEND_DIR}/lib64/libascendcl.so)
set(ACL_TDT_LIB ${ASCEND_DIR}/lib64/libacl_tdt_channel.so)

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/pybind11)
    message(FATAL_ERROR
        "The essential dependency ==> pybind11 <== is not available, "
        "please check existence of path: `mxrec/third_party/pybind11`!"
    )
endif()
set(PYBIND11_DIR ${PROJECT_SOURCE_DIR}/third_party/pybind11)

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/spdlog)
    message(FATAL_ERROR
        "The essential dependency ==> spdlog <== is not available, "
        "please check existence of path: `mxrec/third_party/spdlog`!"
    )
endif()
set(SPDLOG_DIR ${PROJECT_SOURCE_DIR}/third_party/spdlog)

add_subdirectory(host)
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    add_subdirectory(tests)
endif ()
