/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef ACL_CHANNEL_H
#define ACL_CHANNEL_H

#include <vector>
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

