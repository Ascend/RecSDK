# Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
cmake_minimum_required(VERSION 3.5.1)
project(default_gather_pass_pattern1)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

if (DEFINED ENV{ASCEND_CUSTOM_OPP_PATH})
    set(ASCEND_CUSTOM_OPP_PATH $ENV{ASCEND_CUSTOM_OPP_PATH})
else ()
    set(ASCEND_CUSTOM_OPP_PATH "")
    message(WARNING "未定义环境变量 ASCEND_CUSTOM_OPP_PATH，将跳过该路径的include")
endif()

set(ASCEND_CUSTOM_OPP_INCLUDE_PATHS "")  
if (ASCEND_CUSTOM_OPP_PATH)
    string(REPLACE ":" ";" ASCEND_CUSTOM_OPP_PATH_LIST ${ASCEND_CUSTOM_OPP_PATH})
    
    foreach(SINGLE_PATH ${ASCEND_CUSTOM_OPP_PATH_LIST})
        string(STRIP ${SINGLE_PATH} SINGLE_PATH_STRIPPED)
        
        set(CUSTOM_OPP_INCLUDE_PATH "${SINGLE_PATH_STRIPPED}/op_proto")
        
        if (IS_DIRECTORY ${CUSTOM_OPP_INCLUDE_PATH})
            list(APPEND ASCEND_CUSTOM_OPP_INCLUDE_PATHS ${CUSTOM_OPP_INCLUDE_PATH})
        else ()
            message(WARNING "路径不存在，跳过：${CUSTOM_OPP_INCLUDE_PATH}")
        endif()
    endforeach()
endif()

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

set(common_compile_options
    -std=c++11
    -g
    -Wall
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0
)

############ libfuse_tile_concat_pass.so ############
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

set_target_properties(${PROJECT_NAME} PROPERTIES output_name ${PROJECT_NAME})

target_compile_options(${PROJECT_NAME} PRIVATE
    ${common_compile_options}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${common_compile_definitions}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
    ${ASCEND_PATH}/include/graph
    ${ASCEND_PATH}/include/ge
    ${ASCEND_PATH}/include/register
    ${ASCEND_PATH}/include
    ${ASCEND_PATH}/opp/vendors/customize/op_proto
    ${ASCEND_CUSTOM_OPP_INCLUDE_PATHS}
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/compiler/lib64
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    -Wl,--no-as-needed
    graph
    register
    -Wl,--as-needed
)
