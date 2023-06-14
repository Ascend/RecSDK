/*
 * @Copyright: Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * @Description:
 * @Version: 1.0
 * @Author: dev
 * @Date: 2023-05-5 09:50:00
 * @LastEditors: dev
 * @LastEditTime: 2023-05-5 09:50:00
 */

#ifndef UNIQUE_OCK_CTR_COMMON_H
#define UNIQUE_OCK_CTR_COMMON_H

#include <cstdint>
#include <string>
#include <memory>
#include "unique.h"


#ifdef __cplusplus
extern "C" {
#endif

using ExternalLog = void (*)(int level, const char *msg);

#ifdef __cplusplus
}
#endif

#include "ock_ctr_common_def.h"

namespace ock {
namespace ctr {
class Factory;

using FactoryPtr = std::shared_ptr<Factory>;
using UniquePtr = std::shared_ptr<Unique>;

class Factory {
public:
    virtual ~Factory() = default;
    virtual int CreateUnique(UniquePtr &out) = 0;
    virtual int SetExternalLogFuncInner(ExternalLog logFunc) = 0;

public:
    static int Create(FactoryPtr &out)
    {
        int result = 0;
        uintptr_t factory = 0;
        /* dynamic load function */
        if ((result = OckCtrCommonDef::CreatFactory(&factory)) == 0) {
            out.reset(reinterpret_cast<Factory *>(factory));
        }
        return result;
    }
};
}
}

#endif // UNIQUE_OCK_CTR_COMMON_H
