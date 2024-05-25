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

set(OCK_ACCCTR_LINK_FLAGS -Wl,-z,relro,-z,now,-z,noexecstack -s)

set(OCK_CTR_SRC_DIR                 ${PROJECT_SOURCE_DIR})
set(OCK_CTR_SRC_INCLUDE_DIR         ${PROJECT_SOURCE_DIR}/src/include)
set(OCK_CTR_COMMON_DIR              ${PROJECT_SOURCE_DIR}/src/common)
set(OCK_CTR_BUILD_PATH              ${PROJECT_SOURCE_DIR}/build)

set(OUTPUT                          ${PROJECT_SOURCE_DIR}/output)
set(OCK_CTR_PLATFORM_UTIL_DIR       ${PROJECT_SOURCE_DIR}/../../../opensource)
set(OCK_CTR_UTIL_INSTALL_DIR        ${PROJECT_SOURCE_DIR}/install)

if (${BUILD_MODE} MATCHES "ut")
    add_compile_options(-ftest-coverage -fprofile-arcs)
    link_libraries(gcov)
else()
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)  # must set this option otherwise pybind will not find embCache symbol
endif (${BUILD_MODE} MATCHES "ut")

if (${BUILD_MODE} MATCHES "fuzz")
    add_compile_options(-ftest-coverage -fprofile-arcs -fdump-rtl-expand)
    link_libraries(gcov asan)
endif (${BUILD_MODE} MATCHES "fuzz")

message("include : " ${OCK_CTR_SRC_INCLUDE_DIR})

set(LIB_HW_SECURE ${OCK_CTR_PLATFORM_UTIL_DIR}/securec/lib/libsecurec.so)

add_subdirectory(include)
add_subdirectory(common)
add_subdirectory(unique)
add_subdirectory(embedding_cache)


file(GLOB_RECURSE CTR_SRC factory_impl.cpp)

add_library(_ock_ctr_common SHARED ${CTR_SRC})

target_include_directories(_ock_ctr_common
        PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${OCK_CTR_SRC_INCLUDE_DIR}
        ${OCK_CTR_COMMON_DIR}/util)

target_link_libraries(_ock_ctr_common PUBLIC
        -Wl,--start-group
        unique
        embedding_cache
        dl
        utils
        ${LIB_HW_SECURE}
        ${OCK_ACCCTR_LINK_FLAGS}
        -Wl,--end-group)

set(TARGET_INSTALL_LIB ${OUTPUT}/ock_ctr_common/lib)
install(TARGETS _ock_ctr_common DESTINATION ${TARGET_INSTALL_LIB}/ PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)


