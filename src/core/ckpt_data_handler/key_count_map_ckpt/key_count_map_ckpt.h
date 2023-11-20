/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2023-11-01
 */

#ifndef MXREC_KEY_COUNT_MAP_CKPT_H
#define MXREC_KEY_COUNT_MAP_CKPT_H

#include "ckpt_data_handler/ckpt_data_handler.h"

namespace MxRec {
    using namespace std;

    class KeyCountMapCkpt : public CkptDataHandler {
    public:
        KeyCountMapCkpt() = default;
        ~KeyCountMapCkpt() override {}

        void SetProcessData(CkptData& processData) override;
        void GetProcessData(CkptData& processData) override;

        vector<CkptDataType> GetDataTypes() override;

        vector<string> GetDirNames() override;
        vector<string> GetEmbNames() override;
        CkptTransData GetDataset(CkptDataType dataType, string embName) override;

        void SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData) override;

    private:
        const vector<string> fileDirNames { "HashTable", "FEAT_INFO" };
        const vector<CkptDataType> saveDataTypes { CkptDataType::KEY_COUNT_MAP };

        const int embHashElmtNum = 2;

        KeyCountMemT saveKeyCountMap;
        KeyCountMemT loadKeyCountMap;
    };
}

#endif // MXREC_KEY_COUNT_MAP_CKPT_H
