/*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
* Description: acl channel api
* Author: MindX SDK
* Date: 2022/11/15
*/

#ifndef ACL_CHANNEL_H
#define ACL_CHANNEL_H

#include <vector>
#include <string>
#include "acl/acl_tdt.h"
#include "tensorflow/core/framework/tensor.h"


namespace tensorflow {
#ifdef CANN5_x
    Status RecvTensorByAcl(acltdtChannelHandle *acl_handle, std::vector<Tensor> &tensors);
#else

    Status RecvTensorByAcl(const acltdtChannelHandle* aclHandle, std::vector<Tensor>& tensors);


#endif

    Status SendTensorsByAcl(const acltdtChannelHandle* aclHandle, acltdtTensorType aclType,
                            const std::vector<Tensor>& tensors, bool& isNeedResend);

}  // namespace tensorflow

#endif  // ACL_CHANNEL_H

