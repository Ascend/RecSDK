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

#ifndef OCK_UNIQUE_IMPL_H
#define OCK_UNIQUE_IMPL_H

#include "unique_func.h"
#include "factory.h"
namespace ock {
namespace ctr {
class UniqueImpl : public Unique {
public:
    explicit UniqueImpl();
    ~UniqueImpl() override = default;

public:
    int Initialize(const UniqueConf &conf) override;
    void UnInitialize() override;
    int DoUnique(UniqueIn &uniqueIn, UniqueOut &uniqueOut) override;
    int DoEnhancedUnique(UniqueIn &uniqueIn, EnhancedUniqueOut &uniqueOut) override;
    int SetExternalThreadFuncInner(ExternalThread threadFunc) override;

private:
    int CheckInput(UniqueIn &uniqueIn, UniqueOut &uniqueOut);
    bool CheckInputNull(void *ptr, const std::string &name);
    bool CheckInputZero(int64_t in, const std::string &name);
    bool IsInitialized();
    int CheckConf(const UniqueConf &conf);
    int CheckInput(UniqueIn &uniqueIn, EnhancedUniqueOut &uniqueOut);
    int CheckNormalConf(const UniqueConf &conf);
    int CheckEnhancedUniqueConf(const UniqueConf &conf);

private:
    ShardedDedup *unique = nullptr;
    UniqueConf uniqueConf{};
};
}
}

#endif // OCK_UNIQUE_IMPL_H
