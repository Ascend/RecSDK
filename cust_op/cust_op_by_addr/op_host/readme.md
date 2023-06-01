# tik2 op_host


### 目录结构

```
├── op_host                  //算子原型&tiling
│   ├── embedding_update_by_address_tiling.h
│   ├── embedding_update_by_address.cpp
│   ├── CMakeLists.txt        
```


### tiling.h 文件编写

embedding_update_by_address_tiling.h

此处定义了一个结构体

这些定义的变量通常的作用是将 输入shape、attr值等数据信息进行处理后传入算子kerenl中
```
#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(TilingData2)
    TILING_DATA_FIELD_DEF(int32_t, update_dim);
    TILING_DATA_FIELD_DEF(int32_t, addr_nums);
    TILING_DATA_FIELD_DEF(int32_t, ub_limit);
    TILING_DATA_FIELD_DEF(int32_t, embbeding_type);
    TILING_DATA_FIELD_DEF(int32_t, update_type);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(EmbeddingUpdateByAddress, TilingData2)
}
```

### cpp 文件编写

##### 整体结构

```
static ge::graphStatus TilingFunc(gert::TilingContext *context)         //计算tiling的变量值

ge::graphStatus InferShape(gert::InferShapeContext *context)            //设置输出的shape

ge::graphStatus InferShapeRange(gert::InferShapeRangeContext *context)  //设置输出的shaperange

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)      //设置输出的dtype

class EmbeddingUpdateByAddress : public OpDef                           //此处用来定义算子的信息库
```

##### 通用代码

```
//获取输入输出 shape
gert::Shape *x_shape = context->GetInputShape(0)
gert::Shape *y_shape = context->GetInputShape(1)
gert::Shape *z_shape = context->GetOutputShape(0)

std::vector<int64_t> x_dims = x_shape->GetDims(); // shape的dims x_dims = {232,123,2} 
size_t x_dims_len = x_shape->GetDimNum();         // shape的维度 x_dims_len = 3
int64_t x_dim2 = x_shape->GetDim(1);              // shape的第二维  x_dim2 = 123
int64_t x_shapesize = x_shape->GetDimNum();       // shape总大小 x_shapesize = 232*123*2   

//设置shape
*z_shape=Shape(x_dims);    //将z_shape 设置为 {232,123,2} 
*z_shape=*x_shape;         //将z_shape 和x_shape保持一致


//获取与设置  数据类型
DataType input2_dtype = context->GetInputDataType(1);  //获取第二个输入的 DataType
context->SetOutputDataType(0, input2_dtype);
context->SetOutputDataType(1, ge::DataType(DT_FLOAT));


//获取 attr
auto *attrs = context->GetAttrs();
const auto attr0_value = *(attrs->GetAttrPointer<int64_t>(0));  // attr0_value = 算子第1个属性的值
const auto attr1_value = *(attrs->GetAttrPointer<int64_t>(1));  // attr1_value = 算子第2个属性的值
```




##### TilingFunc 编写

```
TilingData1 tiling;

int32_t input_shape = context->GetInputTensor(0)->GetShapeSize(); //获取输入大小
tiling.set_addr_nums(input_shape);   //设置 tiling的addr_nums ，addr_nums=input_shape
tiling.set_xxx(input_shape);         //xxx 对应为tiling头文件所设置的变量名
```

#### 原型 编写

```
namespace ge
{
    ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        gert::Shape *y_shape = context->GetOutputShape(0);
        int64_t input_shape = context->GetInputTensor(0)->GetShapeSize();
        int64_t input_dim = context->GetInputTensor(1)->GetShapeSize() / input_shape;
        y_shape->SetDimNum(2);
        y_shape->SetDim(0, input_shape);
        y_shape->SetDim(1, input_dim);
        return GRAPH_SUCCESS;
    }
    ge::graphStatus InferShapeRange(gert::InferShapeRangeContext *context)
    {
        return GRAPH_SUCCESS;
    }
    ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        context->SetOutputDataType(0, ge::DataType(DT_FLOAT));
        return GRAPH_SUCCESS;
    }
}
```

#### 信息库编写

```
namespace ops
{
    class EmbeddingUpdateByAddress : public OpDef
    {
    public:
        EmbeddingUpdateByAddress(const char *name) : OpDef(name)
        {
            this->Input("address")                                                      //设置算子第1个输入
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("embedding")                                                    //设置算子第2个输入
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")                                                           //设置算子第1个输出
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("update_type").AttrType(OPTIONAL).Int(0);                        //设置算子第1个属性，默认值 0

            this->SetInferShape(ge::InferShape)                                         // 原型推导 调用
                .SetInferDataType(ge::InferDataType);
                //.SetInferShapeRange(ge::InferShapeRange);                             //如果没有设置，则不用调用

            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .SetTilingParse(optiling::TilingPrepare)
                .SetCheckSupport(optiling::check_op_support);

            OpAICoreConfig aicConfig;
            aicConfig.AsyncFlag(true)
                .DynamicCompileStaticFlag(true)                       
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .PrecisionReduceFlag(false)                           //允许上层框架混合精度（fp32->fp16）
                .RangeLimitValue("limited");
            this->AICore().AddConfig("ascend910b", aicConfig);        //设置支持910b
            this->AICore().AddConfig("ascend910", aicConfig);         //增加支持910
        }
    };
    OP_ADD(EmbeddingUpdateByAddress, optiling::TilingCompileInfo);
}

```



### tips 

#### 数据类型参考
```
    enum DataType {
        DT_FLOAT = 0,           // float type
        DT_FLOAT16 = 1,         // fp16 type
        DT_INT8 = 2,            // int8 type
        DT_INT16 = 6,     // int16 type
        DT_UINT16 = 7,      // uint16 type
        DT_UINT8 = 4,           // uint8 type
        DT_INT32 = 3,           //
        DT_INT64 = 9,           // int64 type
        DT_UINT32 = 8,          // unsigned int32
        DT_UINT64 = 10,          // unsigned int64
        DT_BOOL = 12,            // bool type
        DT_DOUBLE = 11,          // double type
        DT_STRING = 13,            // string type
        DT_DUAL_SUB_INT8 = 14,    /**< dual output int8 type */
        DT_DUAL_SUB_UINT8 = 15,    /**< dual output uint8 type */
        DT_COMPLEX64 = 16,         // complex64 type
        DT_COMPLEX128 = 17,        // complex128 type
        DT_QINT8 = 18,             // qint8 type
        DT_QINT16 = 19,            // qint16 type
        DT_QINT32 = 20,            // qint32 type
        DT_QUINT8 = 21,            // quint8 type
        DT_QUINT16 = 22,           // quint16 type
        DT_RESOURCE = 23,          // resource type
        DT_STRING_REF = 24,        // string ref type
        DT_DUAL = 25,              // dual output type
        DT_VARIANT = 26,           // dt_variant type
        DT_BF16 = 27,              // bf16 type
        DT_UNDEFINED = 28,         // Used to indicate a DataType field has not been set.
        DT_INT4 = 29,              // int4 type
        DT_MAX                     // Mark the boundaries of data types
    };

```

