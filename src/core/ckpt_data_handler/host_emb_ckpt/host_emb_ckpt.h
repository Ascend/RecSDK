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
        EmbMemT* saveHostEmbs;
        EmbMemT* loadHostEmbs;

        void SetEmbInfoTrans(string embName);
        void SetEmbDataTrans(string embName);

        void SetEmbInfo(string embName, CkptData& ckptData);
        void SetEmbData(string embName, CkptData& ckptData) const;

        int GetEmbInfoSize();
        size_t GetEmbDataSize(string embName);
        size_t GetEmbDataRows(string embName);
    };
}

#endif // MX_REC_HOST_EMB_CKPT_H
