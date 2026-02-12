/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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
#ifndef ARGS_STRUCT_H
#define ARGS_STRUCT_H

#include "kernel_operator.h"
using namespace AscendC;

struct Args {
    GM_ADDR gradOutput;
    GM_ADDR devWeights;
    GM_ADDR weightsPlacements;
    GM_ADDR weightsOffsets;
    GM_ADDR dOffsets;
    GM_ADDR hashSizeCumsum;
    GM_ADDR indices;
    GM_ADDR offsets;
    GM_ADDR momentum1Dev;
    GM_ADDR momentum2Dev;
    GM_ADDR hashIndices;
    GM_ADDR uniqueId;
    GM_ADDR uniqueHashSize;
    GM_ADDR uniqueInverse;
    GM_ADDR indiceSizeCumsum;

    GM_ADDR out;
    GM_ADDR momentum1DevOut;
    GM_ADDR momentum2DevOut;
    GM_ADDR weightsDevOut;

    GM_ADDR workspace;
    GM_ADDR tiling;
};

#endif // ARGS_STRUCT_H