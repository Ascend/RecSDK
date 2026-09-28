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

echo "安装clangtidy"
python3 -m pip install clang-tidy

python3 -m pip install --upgrade pre-commit
echo "===== precommit ====="
python3 -m pre_commit --version
cd ./${serviceName}
echo "待检测C/C++源文件总数：${#CHECK_ARRAY[@]}"
for single_file in "${CHECK_ARRAY[@]}"; do
    echo "----------------------------------------"
    ls -a
    echo "正在检测文件：${single_file}"
    cat ${single_file}

    res=$(find -name "compile_commands.json")
    if [ -z "$res" ];then
        echo "WARNING: 未找到 compile_commands.json，clang-tidy 可能头文件查找失败"
    else
        echo "找到编译数据库：$res"
    fi
    python3 -m pre_commit run clang-tidy --files "${single_file}" </dev/null
    ret=$?
    echo "查看返回码：${ret}"
    if [[ ${ret} -ne 0 ]]; then
        echo "ERROR: 文件 ${single_file} clang-tidy 检测不通过"
        exit ${ret}
    fi
done
echo "===== 全部源文件clang-tidy检测完成 ====="
