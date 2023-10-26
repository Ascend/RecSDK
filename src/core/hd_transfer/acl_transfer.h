//
// Created by w00842226 on 2023/10/26.
//
#include "hd_transfer.h"
#include <fstream>
#include "utils/common.h"
#include "utils/time_cost.h"

using namespace MxRec;
using namespace std;
#ifndef MXREC_ACL_TRANSFER_H
#define MXREC_ACL_TRANSFER_H
enum class AclTransferStatus {
    OK,
    F001
};
AclTransferStatus RecvByAcl(const acltdtChannelHandle *handle, const acltdtDataset *dataset, float* resultPtr);

#endif //MXREC_ACL_TRANSFER_H
