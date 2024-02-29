/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 ==============================================================================*/

#ifndef OCK_FACTORY_IMPL_H
#define OCK_FACTORY_IMPL_H

#include "include/factory.h"
#include "unique/unique_impl.h"

namespace ock {
namespace ctr {
class FactoryImpl : public Factory {
public:
    FactoryImpl() = default;
    ~FactoryImpl() override = default;

public:
    int CreateUnique(std::shared_ptr<Unique> &out) override;
    int SetExternalLogFuncInner(ExternalLog logFunc) override;

public:
    static Factory *gGlobalFactory;
    static Lock gLock;
};
}
}

#endif // OCK_FACTORY_IMPL_H