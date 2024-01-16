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
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

file(GLOB_RECURSE MXREC_OP_EOS_DATA ./*.cc)
add_library(rec_eos_ops SHARED ${MXREC_OP_EOS_DATA})

set(TF_INTERNAL_LIB ${TF_PATH}/python/_pywrap_tensorflow_internal.so)

target_link_libraries(rec_eos_ops PUBLIC ASC
        ${TF_INTERNAL_LIB}
        )


install(TARGETS rec_eos_ops LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})