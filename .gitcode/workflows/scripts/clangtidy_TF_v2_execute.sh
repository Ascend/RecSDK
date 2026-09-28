#!/bin/bash
set -e
serviceName=$1


# 读取变更文件列表
MODIFY_FILE="change.txt"
FILE_ARRAY=()
exec 3< "${MODIFY_FILE}"
while IFS= read -r raw_line <&3 || [[ -n "${raw_line}" ]]; do
    fp=$(echo "${raw_line}" | sed -e 's/^"//' -e 's/"$//' -e 's/^[ \t]*//' -e 's/[ \t]*$//')
    if [[ -n "${fp}" ]]; then
        FILE_ARRAY+=("${fp}")
    fi
done
exec 3<&-

# 过滤只保留源码文件传入检测
CHECK_ARRAY=()
for file in "${FILE_ARRAY[@]}"; do
    if [[ "${file}" == *training/torch_rec_v2* ]] && [[ "${file}" =~ [.](cpp|cc|cxx|c)$ ]]; then
        CHECK_ARRAY+=("${file}")
    fi
done

if [[ ${#CHECK_ARRAY[@]} -eq 0 ]]; then
    echo "INFO: 本次变更没有 .cpp / .cc / .cxx / .c 文件，无需代码检查，脚本退出"
    exit 0
fi

echo "===== 开始安装clang与pre-commit并执行静态检测 ====="

cd ./${serviceName}/build && dos2unix *.sh && chmod +x *

bash clang_tidy_check.sh tf_v2

echo "===== 全部源文件clang-tidy检测完成 ====="
