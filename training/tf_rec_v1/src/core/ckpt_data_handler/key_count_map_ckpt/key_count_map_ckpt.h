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
