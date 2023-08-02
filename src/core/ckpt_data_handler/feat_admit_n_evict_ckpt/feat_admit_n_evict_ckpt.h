/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-22
 */

#ifndef MXREC_FEAT_ADMIT_N_EVICT_CKPT_H
#define MXREC_FEAT_ADMIT_N_EVICT_CKPT_H

#include "ckpt_data_handler/ckpt_data_handler.h"

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
        const vector<string> fileDirNames { "HashTable", "DDR" };
        const vector<CkptDataType> saveDataTypes { CkptDataType::TENSOR_2_THRESH, CkptDataType::HIST_REC };

        const int featItemInfoSaveNum { 3 };
        const int threshValSaveNum { 2 };

        const int countThresholdIdx { 0 };
        const int timeThresholdIdx { 1 };

        const int featItemInfoOffset { 1 };

        const int featureIdIdxOffset { 0 };
        const int countIdxOffset { 1 };
        const int lastTimeIdxOffset { 2 };

        tensor_2_thresh_mem_t saveTens2Thresh;
        tensor_2_thresh_mem_t loadTens2Thresh;

        AdmitAndEvictData saveHistRec;
        AdmitAndEvictData loadHistRec;

        void ClearData();

        void SetTens2ThreshTrans(string embName);
        void SetHistRecTrans(string embName);

        void SetTens2Thresh(string embName);
        void SetHistRec(string embName);

        int GetTens2ThreshSize();
        size_t GetHistRecSize(string embName);
    };
}

#endif // MXREC_FEAT_ADMIT_N_EVICT_CKPT_H
