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

add_compile_options(-ftest-coverage -fprofile-arcs)
link_libraries(gcov)

set(OCK_CTR_UTIL_INSTALL_DIR        ${PROJECT_SOURCE_DIR}/install)
set(OCK_CTR_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
message("src" ${OCK_CTR_SRC_DIR})

file(GLOB_RECURSE TEST_UNIQUE_FILES *.cpp *.h)
add_executable(test_unique_files ${TEST_UNIQUE_FILES})
include_directories(${OCK_CTR_UTIL_INSTALL_DIR}/googletest-release-1.8.1/include)
link_directories(${OCK_CTR_UTIL_INSTALL_DIR}/googletest-release-1.8.1/lib64)

SET(LIB_3RD_GMOCK ${OCK_CTR_UTIL_INSTALL_DIR}/googletest-release-1.8.1/lib64/libgmock.a)
SET(LIB_3RD_GTEST ${OCK_CTR_UTIL_INSTALL_DIR}/googletest-release-1.8.1/lib64/libgtest.a)


message(${OCK_CTR_SRC_DIR}/include)

target_include_directories(test_unique_files
        PUBLIC
        ${OCK_CTR_SRC_DIR}/include)

target_link_libraries(test_unique_files
        PUBLIC
        -Wl,--start-group
        pthread
        dl
        ${LIB_3RD_GTEST}
        ${LIB_3RD_GMOCK}
        -Wl,--end-group)

