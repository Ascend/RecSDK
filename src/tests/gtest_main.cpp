/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: gtest main
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include <mpi.h>
#include <gtest/gtest.h>

int main(int argc, char *argv[])
{
    int result = 0;
    ::testing::InitGoogleTest(&argc, argv);
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}