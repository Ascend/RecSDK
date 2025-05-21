#ifndef TILING_POLICY_NORMAL_V200_FUXI_H
#define TILING_POLICY_NORMAL_V200_FUXI_H

#include "tiling_policy.h"

namespace HstuDenseForwardFuxi {

class TilingPolicyNormalv200Fuxi : public TilingPolicy {
public:
    bool GeneralShapeCheck(int64_t batchSize, int64_t seqLen, int64_t headNum, int64_t dim) override;
private:
    ge::graphStatus InferShape(gert::InferShapeContext* context) override;
    bool TilingShape(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling) override;
    bool TilingHeighLevelApi(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling) override;
    bool TilingKeySet(gert::TilingContext* context, optiling::HstuDenseForwardFuxiTilingData &tiling) override;
};
}

#endif