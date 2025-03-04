/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RMA_LOG_H
#define RMA_LOG_H

#include <string>
#include <cstdlib>
namespace optiling{
constexpr int32_t DEBUG_SWITCH = 0;

#if DEBUG_SWITCH
#define HSHMEM_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOG_HSHMEM(_msg, ...) printf("[%s:%4d] " _msg "\n", HSHMEM_FILENAME, __LINE__, ##__VA_ARGS__)
#define LOG_ERROR(_msg, ...) LOG_HSHMEM("[ERRNO] " _msg, ##__VA_ARGS__)
#define LOG_INFO(_msg, ...) LOG_HSHMEM("[INFO] " _msg, ##__VA_ARGS__)
#define LOG_DEBUG(_msg, ...) LOG_HSHMEM("[DEBUG] " _msg, ##__VA_ARGS__)

#else
#define LOG_ERROR(_msg, ...)
#define LOG_INFO(_msg, ...)
#define LOG_DEBUG(_msg, ...)
#endif
}
#endif