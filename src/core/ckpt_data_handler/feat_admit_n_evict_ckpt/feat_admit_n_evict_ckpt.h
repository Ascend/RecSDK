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

#ifndef MXREC_FEAT_ADMIT_N_EVICT_CKPT_H
#define MXREC_FEAT_ADMIT_N_EVICT_CKPT_H

#include "ckpt_data_handler/ckpt_data_handler.h"
#include "utils/config.h"

namespace MxRec {
    using namespace std;

    class FeatAdmitNEvictCkpt : public CkptDataHandler {
    public:
        FeatAdmitNEvictCkpt() = default;
        ~FeatAdmitNEvictCkpt() override {}

        void SetProcessData(CkptData& processData) override;
        void GetProcessData(CkptData& processData) override;

        vector<CkptDataType> GetDataTypes() override;

        vector<string> GetDirNames() override;
        vector<string> GetEmbNames() override;
        CkptTransData GetDataset(CkptDataType dataType, string embName) override;

        void SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData) override;

    private:
        const vector<string> fileDirNames { "HashTable", "FEAT_INFO" };
        const vector<CkptDataType> saveDataTypes { CkptDataType::TABLE_2_THRESH, CkptDataType::HIST_REC };

        const int featItemInfoSaveNum { 3 };
        const int threshValSaveNum { 3 };

        const int countThresholdIdx { 0 };
        const int timeThresholdIdx { 1 };
        const int isSumThresholdIdx { 2 };

        const int featItemInfoOffset { 1 };

        const int featureIdIdxOffset { 0 };
        const int countIdxOffset { 1 };
        const int lastTimeIdxOffset { 2 };

        Table2ThreshMemT saveTable2Thresh;
        Table2ThreshMemT loadTable2Thresh;

        AdmitAndEvictData saveHistRec;
        AdmitAndEvictData loadHistRec;

        void ClearData();

        void SetTable2ThreshTrans(string embName);
        void SetHistRecTrans(string embName);

        void SetTable2Thresh(string embName);
        void SetHistRec(string embName);

        int GetTable2ThreshSize();
        size_t GetHistRecSize(string embName);
    };
}

#endif // MXREC_FEAT_ADMIT_N_EVICT_CKPT_H
