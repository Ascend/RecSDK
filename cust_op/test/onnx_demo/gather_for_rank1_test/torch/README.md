## gather_for_rank1算子适配onnx框架说明

### 约束
Atlas 推理系列产品

### 1.编译算子工程
执行算子run包进行算子安装部署

### 2.转模型
  - 生成onnx模型（已有模型请跳过）
    ```bash
    python3 gather.py
    ```
  - 进入到存放onnx模型的目录, 进行模型转换，可参考命令：
    ```bash
    atc --model=./gather.onnx --framework=5 --output=gather --soc_version=Ascend310P3 --input_format=ND
    ```

### 3.模型推理
  - 准备符合模型输入的测试数据，使用(msame)[https://gitee.com/ascend/tools/tree/master/msame]进行模型推理
    ```bash
    ./msame --model "gather.om" --input "input/x.bin,input/index.bin" --output "output" --outfmt BIN
    ```