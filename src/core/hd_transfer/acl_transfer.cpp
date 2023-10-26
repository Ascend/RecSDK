//
// Created by w00842226 on 2023/10/26.
//
#include "acl_transfer.h"

AclTransferStatus RecvByAcl(const acltdtChannelHandle *handle, acltdtDataset *dataset, float* resultPtr){
#ifndef GTEST
    if (dataset==nullptr || handle==nullptr) {
        throw runtime_error(StringFormat("handle or dataset is nullptr:%s.");
    }
    auto aclStatus = acltdtReceiveTensor(handle, dataset, GlobalEnv::aclTimeout);
    if (aclStatus != ACL_ERROR_NONE && aclStatus != ACL_ERROR_RT_QUEUE_EMPTY) {
        return AclTransferStatus::F001;
    }
    auto size = acltdtGetDatasetSize(dataset);
    if (size == 0) {
        LOG_WARN(HOSTEMB + "recv empty data");
        return AclTransferStatus::OK;
    }
    auto aclData = acltdtGetDataItem(dataset, 0);
    if (aclData == nullptr) {
        return AclTransferStatus::F001;
    }
    resultPtr = reinterpret_cast<float *>(acltdtGetDataAddrFromItem(aclData));
#endif
    return AclTransferStatus::OK;;
}