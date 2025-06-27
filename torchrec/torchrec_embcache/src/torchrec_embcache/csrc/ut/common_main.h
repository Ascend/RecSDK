/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef COMMON_MAIN_H
#define COMMON_MAIN_H
#include <gtest/gtest.h>
#include <glog/logging.h>

int CommonMain(int argc, char* argv[])
{
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);

    ::testing::InitGoogleTest(&argc, argv);

    int result = 0;
    result = RUN_ALL_TESTS();
    return result;
}
#endif // COMMON_MAIN_H