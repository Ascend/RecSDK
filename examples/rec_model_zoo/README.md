# EffBench

该文档介绍如何在NPU进行EffBench的profiling。

## 主要依赖

**TensorFlow:** 1.15.0

## 数据集准备

1. 对于 [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge) 数据集, 我们采用 [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768) 的预处理，根据其提供的链接下载libsvm格式的数据集，进一步将其转换为tfrecord格式。
2. download url 'https://worksheets.codalab.org/rest/bundles/0x8dca5e7bac42470aa445f9a205d177c6/contents/blob/', 替换到download_criteo.py中的url。
    ```shell
    cd feature_interaction/data/
    python download_criteo.py
    python libsvm2tfrecord.py
    ```

2. 对于 [Ali-CCP](https://tianchi.aliyun.com/dataset/408) 数据集, 我们提供了一套完整的 [预处理流程](behaviour_and_multi_task/data/aliccp/README.md) :
    ```shell
    cd behaviour_and_multi_task/data/aliccp/
    bash run.sh
    ```

## 训练
首先需要在`train.sh`中修改PREPROCESSED_DATASET为数据集的路径。
1. For **feature interaction learning** models:
    ```shell
    cd feature_interaction/src/
    bash train.sh
    ```

2. For **behaviour sequence modeling** and **multi-task learning** models:
    ```shell
    cd behaviour_and_multi_task/src/
    bash train.sh
    ```

## Profiling

Profiling方法参考自[CANN官网](https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/devaids/auxiliarydevtool/atlasprofiling_16_0029.html)。

1. 添加环境变量:
    ```shell
    export PROFILING_MODE=true
    export PROFILING_OPTIONS='{"output":"/home/ma-user/work/ydz/EffBench/feature_interaction/profiling/train","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
    ```

2. 若profiling模型的训练过程，运行`profiling_train.sh`脚本，以特征交互模型为例:
    ```shell
    cd feature_interaction/src/
    bash profiling_train.sh
    # 推理
    # bash profiling_infer.sh
    ```

3. 使用msprof工具解析:
    ```shell
    export MSPORF_TOOL=/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msprof/msprof.py
    
    python $MSPORF_TOOL import -dir <profile的output路径> (比如：/home/ma-user/work/ydz/EffBench/feature_interaction/profiling/train/PROF_000001_20241116222226387_LMAKGCJJFNJORPKC)
    ```

4. 查询性能数据信息，找到最大的`Iteration Number`对应的`Model ID`：
    ```shell
    python $MSPORF_TOOL query -dir <profile的output路径>
    ```

5. 使用msprof工具导出profiling文件:
    ```shell
    python $MSPORF_TOOL export timeline -dir <profile的output路径> --iteration-id 100 --model-id <上一步找到的Model ID>

    python $MSPORF_TOOL export summary -dir <profile的output路径> --iteration-id 100 --model-id <上一步找到的Model ID>
    ```

