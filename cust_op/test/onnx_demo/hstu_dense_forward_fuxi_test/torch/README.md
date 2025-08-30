## hstu_dense_forward算子适配onnx框架说明

### 约束
Atlas 推理系列产品

### 1.算子安装部署
执行算子run包进行算子安装部署

### 2.转模型
  - 生成onnx模型（已有模型请跳过）
    ```bash
    python3 hstu_fuxi.py
    ```
  - 进入到存放onnx模型的目录, 进行模型转换，可参考命令：
    ```bash
    atc --model=./hstu_fuxi.onnx --framework=5 --output=hstu_fuxi --soc_version=Ascend310P3 --input_format=ND --output_type=FP16
    ```

### 3.模型推理
  - 准备符合模型输入的测试数据，使用(msame)[https://gitee.com/ascend/tools/tree/master/msame]进行模型推理
    ```bash
    ./msame --model "hstu_fuxi.om" --input "input/q.bin,input/k.bin,input/v.bin,input/timestamp_bias.bin,input/position_bias.bin,input/mask.bin" --output "output" --outfmt BIN
    ```