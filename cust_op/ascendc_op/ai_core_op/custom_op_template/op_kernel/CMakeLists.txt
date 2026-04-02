# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

npu_op_kernel_options(ascendc_kernels ALL
    OPTIONS "--cce-long-call=true"
)

if (ENABLE_CATLASS AND CATLASS_HOME)
    if (ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
        npu_op_kernel_options(ascendc_kernels ALL
            OPTIONS
                -DCATLASS_ARCH=3510
                -DCATLASS_BISHENG_ARCH=a5
                -DIS_A5=1
                -DENABLE_CV_COMM_VIA_SSBUF=true
                -DCATLASS_HOME=${CATLASS_HOME}
                -I${CATLASS_HOME}/include
        )
    else()
        npu_op_kernel_options(ascendc_kernels ALL
            OPTIONS
                -DCATLASS_ARCH=2201
                -DCATLASS_BISHENG_ARCH=a2
                -DIS_A5=0
                -DENABLE_CV_COMM_VIA_SSBUF=true
                -DCATLASS_HOME=${CATLASS_HOME}
                -I${CATLASS_HOME}/include
        )
    endif()
endif()

npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../${vendor_name}.json
    DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic
)