/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.s
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#pragma once

#include <gmock/gmock.h>

#include "ock_ctr_common/include/factory.h"

namespace ock {
namespace ctr {
class FactoryMock : public Factory {
public:
    MOCK_METHOD1(CreateUnique, int(UniquePtr& out));
    MOCK_METHOD1(CreateEmbCacheManager, int(EmbCacheManagerPtr& out));
    MOCK_METHOD1(SetExternalLogFuncInner, int(ExternalLog logFunc));
};
}  // namespace ctr
}  // namespace ock