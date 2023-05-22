/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-12
 */

#ifndef MX_REC_HOST_EMB_CKPT_H
#define MX_REC_HOST_EMB_CKPT_H

#include "ckpt_data_handler/ckpt_data_handler.h"

namespace MxRec {
    using namespace std;

    class HostEmbCkpt : public CkptDataHandler {
    public:
        HostEmbCkpt() = default;
        ~HostEmbCkpt() override {}

        void SetProcessData(CkptData& processData) override;
        void GetProcessData(CkptData& processData) override;

        vector<CkptDataType> GetDataTypes() override;

        vector<string> GetDirNames() override;
        vector<string> GetEmbNames() override;
        CkptTransData GetDataset(CkptDataType dataType, string embName) override;

        void SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData) override;
        void SetDatasetForLoadEmb(
                CkptDataType dataType, string embName, CkptTransData& loadedData, CkptData& ckptData) override;

    private:
        const vector<string> fileDirNames { "HashTable", "DDR" };
        const vector<CkptDataType> saveDataTypes { CkptDataType::EMB_INFO, CkptDataType::EMB_DATA };

        const int attribEmbInfoSendCntIdx { 0 };
        const int attribEmbInfoEmbSizeIdx { 1 };
        const int attribEmbInfoDevVocabIdx { 2 };
        const int attribEmbInfoHostVocabIdx { 3 };

        const int attribEmbDataOuterIdx { 0 };
        const int attribEmbDataInnerIdx { 1 };

        const int embSveElmtNum { 4 };
        emb_mem_t* saveHostEmbs;
        emb_mem_t* loadHostEmbs;

        void SetEmbInfoTrans(string embName);
        void SetEmbDataTrans(string embName);

        void SetEmbInfo(string embName, CkptData& ckptData);
        void SetEmbData(string embName, CkptData& ckptData);

        int GetEmbInfoSize();
        size_t GetEmbDataSize(string embName);
        size_t GetEmbDataRows(string embName);
    };
}

#endif // MX_REC_HOST_EMB_CKPT_H
