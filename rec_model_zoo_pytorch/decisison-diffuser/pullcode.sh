#!/bin/bash

git clone https://github.com/anuragajay/decision-diffuser.git
cd decision-diffuser
git checkout 01ce528
git apply ../dd.patch

if [ "$?" -eq 0 ]; then
    echo "DD模型patch包已打入，可以进入decision-diffuser/code/analysis目录运行模型"
    echo "请编辑launch.sh后再bash launch.sh命令运行模型"
    echo "注意NPU上需要编辑npu_env.sh环境变量"
else
    echo "执行脚本报错，请重新运行脚本继续打入Patch代码"
fi