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

set(INCLUDE_HEADERS factory.h ock_ctr_common_def.h unique.h embedding_cache.h)

set(TARGET_INSTALL_INCLUDE ${OUTPUT}/ock_ctr_common/include)

install(FILES ${INCLUDE_HEADERS} DESTINATION ${TARGET_INSTALL_INCLUDE} PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ)