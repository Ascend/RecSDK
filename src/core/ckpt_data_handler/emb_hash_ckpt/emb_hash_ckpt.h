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

#ifndef MX_REC_EMB_HASH_CKPT_H
#define MX_REC_EMB_HASH_CKPT_H

#include "ckpt_data_handler/ckpt_data_handler.h"

namespace MxRec {
    using namespace std;

    class EmbHashCkpt : public CkptDataHandler {
    public:
        EmbHashCkpt() = default;
        ~EmbHashCkpt() override {}

        void SetProcessData(CkptData& processData) override;
        void GetProcessData(CkptData& processData) override;

        vector<CkptDataType> GetDataTypes() override;

        vector<string> GetDirNames() override;
        vector<string> GetEmbNames() override;
        CkptTransData GetDataset(CkptDataType dataType, string embName) override;

        void SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData) override;

    private:
        const vector<string> fileDirNames { "HashTable", "DDR" };
        const vector<CkptDataType> saveDataTypes { CkptDataType::EMB_HASHMAP, CkptDataType::DEV_OFFSET,
            CkptDataType::EMB_CURR_STAT, CkptDataType::EVICT_POS };

        const int currUpdataPosIdx { 0 };
        const int hostVocabIdx { 1 };
        const int devVocabIdx { 2 };
        const int maxOffsetIdx { 3 };

        const int attrbDev2BatchIdx { 0 };
        const int attrbDev2KeyIdx { 1 };

        const int attrEvictPosIdx {0};

        const int embHashElmtNum { 2 };
        const int embCurrStatNum { 4 };
        EmbHashMemT saveEmbHashMaps;
        EmbHashMemT loadEmbHashMaps;

        void SetEmbHashMapTrans(string embName);
        void SetDevOffsetTrans(string embName);
        void SetEmbCurrStatTrans(string embName);
        void SetEvictPosTrans(string embName);

        void SetEmbHashMap(string embName);
        void SetDevOffset(string embName);
        void SetEmbCurrStat(string embName);
        void SetEvictPos(string embName);
    };
}

#endif // MX_REC_EMB_HASH_CKPT_H