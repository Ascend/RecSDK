/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-17
 */

#ifndef MX_REC_NDDR_OFFSET_CKPT_H
#define MX_REC_NDDR_OFFSET_CKPT_H

#include "ckpt_data_handler/ckpt_data_handler.h"

namespace MxRec {
    using namespace std;

    class NddrOffsetCkpt : public CkptDataHandler {
    public:
        NddrOffsetCkpt() = default;
        ~NddrOffsetCkpt() override {}

        void SetProcessData(CkptData& processData) override;
        void GetProcessData(CkptData& processData) override;

        vector<CkptDataType> GetDataTypes() override;

        vector<string> GetDirNames() override;
        vector<string> GetEmbNames() override;
        CkptTransData GetDataset(CkptDataType dataType, string embName) override;

        void SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData) override;

    private:
        const vector<string> fileDirNames { "HashTable", "HBM" };
        const vector<CkptDataType> saveDataTypes { CkptDataType::NDDR_OFFSET };

        OffsetMemT saveMaxOffset;
        OffsetMemT loadMaxOffset;
    };
}

#endif // MX_REC_NDDR_OFFSET_CKPT_H
