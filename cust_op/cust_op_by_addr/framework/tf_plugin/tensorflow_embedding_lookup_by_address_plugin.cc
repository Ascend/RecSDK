#include "register/register.h"

namespace domi {
// register op info to GE
REGISTER_CUSTOM_OP("EmbeddingLookupByAddress")
.FrameworkType(TENSORFLOW) // type: CAFFE, TENSORFLOW
.OriginOpType("EmbeddingLookupByAddress") // name in tf module
.ParseParamsByOperatorFn(AutoMappingByOpFn);
REGISTER_CUSTOM_OP("EmbeddingUpdateByAddress")
.FrameworkType(TENSORFLOW) // type: CAFFE, TENSORFLOW
.OriginOpType("EmbeddingUpdateByAddress") // name in tf module
.ParseParamsByOperatorFn(AutoMappingByOpFn);
} // namespace domi