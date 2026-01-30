#!/bin/bash

# 定义错误处理函数：打印失败提示并退出脚本
error_exit() {
    echo "执行失败，请重新执行"
    exit 1
}

# 逐条执行命令，每一步都检查执行状态
echo "开始克隆仓库..."
git clone https://gitee.com/omniai/omniinfer.git || error_exit

echo "进入omniinfer目录..."
cd omniinfer || error_exit

echo "切换到release_v0.6.0分支..."
git checkout release_v0.6.0 || error_exit

echo "应用patch文件..."
git apply ../gr_tp4.patch || error_exit
## 如果git apply报错，那么需要使用dos2unix ../gr_tp4.patch 

# 所有命令执行成功后，打印成功提示
echo "omniinfer的patch已打入"
echo "运行脚本在omniinfer/model_zoo下"