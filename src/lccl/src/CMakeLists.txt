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

set(LCAL_SOURCE_FILE
        lcal_comm.cpp
        tools/socket/lcal_sock_exchange.h
        tools/socket/lcal_sock_exchange.cpp
        ${ATB_LOG_FILES})

add_library(lcal SHARED ${LCAL_SOURCE_FILE})
add_library(lcal_static STATIC ${LCAL_SOURCE_FILE})
set_target_properties(lcal_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(lcal ascendcl runtime profapi c_sec)
target_link_libraries(lcal_static ascendcl runtime profapi c_sec)

install(TARGETS lcal LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS lcal_static DESTINATION ${CMAKE_INSTALL_PREFIX})
