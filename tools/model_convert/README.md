## 模型转换工具使用说明

### 使用背景： 将基于mxRec保存下来的NPU格式的模型，转换为可被GPU、CPU加载使用的模型。

### 使用说明：
示例如下：
`python model_cover.py --input_path=./saved_model --output_path=./saved-model-out --rank_size=8 --estimator=1 --ddr=1`

**参数解释：**

`input_path` : `str`, 必选参数，待转换的模型保存路径
`output_path`: `str`, 必选参数，完成转换的模型输出路径
`rank_size`: `int`, NPU格式的模型训练的卡数，取值范围[1,16],非必选参数，默认值8
`estimator`: `int`, 是否使用estimator训练模型，0-不使用，1-使用。默认值0
`ddr`：`int`, 是否使用ddr模式， 0-不适用， 1-使用。 默认值0
