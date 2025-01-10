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
#include <mpi.h>
#include <vector>
#include <unordered_set>
#include <map>
#include <sstream>
#include <fstream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "lcal_comm.h"

using namespace std;
using namespace Lcal;


TEST(LcclTest, Initialization)
{
    std::cout << "===========Initialization start=============" << std::endl;
    int rank;
    int rankSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
    aclError aclRet = aclrtSetDevice(rank);
    Lcal::LcalComm c(rank, rankSize);
    auto ret = c.Init();
    ASSERT_EQ(ret, 0);
    std::cout << "===========Initialization end=============" << std::endl;
}