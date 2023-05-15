/*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
* Description: acl channel api
* Author: MindX SDK
* Date: 2022/11/15
*/

#ifndef ACL_CHANNEL_H_
#define ACL_CHANNEL_H_

#include <vector>
#include <string>
#include "acl/acl_tdt.h"
#include "tensorflow/core/framework/tensor.h"


namespace tensorflow {
#ifdef CANN5_x
    Status RecvTensorByAcl(acltdtChannelHandle *acl_handle, std::vector<Tensor> &tensors);
#else

    Status RecvTensorByAcl(const acltdtChannelHandle* acl_handle, std::vector<Tensor>& tensors);

    Status StopRecvTensorByAcl(acltdtChannelHandle **handle, const std::string &channel_name);

#endif

    Status SendTensorsByAcl(const acltdtChannelHandle* acl_handle, acltdtTensorType acl_type,
                            const std::vector<Tensor>& tensors, bool& is_need_resend);

}  // namespace tensorflow

#endif  // ACL_CHANNEL_H_

