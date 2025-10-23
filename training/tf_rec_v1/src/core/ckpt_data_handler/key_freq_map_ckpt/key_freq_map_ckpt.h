/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#ifndef MX_REC_KEY_FREQ_MAP_CKPT_H
#define MX_REC_KEY_FREQ_MAP_CKPT_H

#include "ckpt_data_handler/ckpt_data_handler.h"

namespace MxRec {
    using namespace std;

    class KeyFreqMapCkpt : public CkptDataHandler {
    public:
        KeyFreqMapCkpt() = default;
        ~KeyFreqMapCkpt() override {}

        void SetProcessData(CkptData& processData) override;
        void GetProcessData(CkptData& processData) override;

        vector<CkptDataType> GetDataTypes() override;

        vector<string> GetDirNames() override;
        vector<string> GetEmbNames() override;
        CkptTransData GetDataset(CkptDataType dataType, string embName) override;

        void SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData) override;

    private:
        const vector<string> fileDirNames { "HashTable", "SSD" };
        const vector<CkptDataType> saveDataTypes { CkptDataType::DDR_FREQ_MAP, CkptDataType::EXCLUDE_FREQ_MAP};

        const int freqMapElmtNum { 2 }; // Number of element types in the keyFreqMap during saving

        KeyFreqMemT saveDDRKeyFreqMaps;
        KeyFreqMemT loadDDRKeyFreqMaps;
        KeyFreqMemT saveExcludeDDRKeyFreqMaps;
        KeyFreqMemT loadExcludeDDRKeyFreqMaps;

        void SetDDRFreqMapTrans(string embName);
        void SetExcludeDDRFreqMapTrans(string embName);

        void SetDDRFreqMaps(string embName);
        void SetExcludeDDRFreqMaps(string embName);
    };
}

#endif // MX_REC_KEY_FREQ_MAP_CKPT_H
