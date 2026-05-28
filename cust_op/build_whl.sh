#!/bin/bash
set -e

# 清理历史缓存
rm -rf build/
rm -rf _skbuild/
rm -rf rec_sdk_ops.egg-info/
rm -rf dist/

# CATLASS 准备逻辑已移除。
# 预留说明：后续统一通过 submodule 管理，请在构建前执行：
#   git submodule update --init --recursive

# AscendC 算子编译范围（受 CANN 平台信息限制）
# 例：在 A2 机器上构建全量包:  RECSDK_BUILD_VERS=A2,A3 bash build_whl.sh
# torch_plugin 适配层 .so 始终编译全部变体（A5 + A2A3），不受此变量影响
# 单一入口：默认全编译，必要时可在命令前显式覆盖 RECSDK_BUILD_VERS。
export RECSDK_BUILD_VERS="${RECSDK_BUILD_VERS:-A2,A3,A5}"

# AscendC 编译串并行开关：
# - 外部入口（推荐）：SERIAL_BUILD=ON|OFF bash build_whl.sh
# - 内部变量（兼容）：RECSDK_ASCEND_SERIAL_BUILD=ON|OFF
# 优先级：SERIAL_BUILD > RECSDK_ASCEND_SERIAL_BUILD > 默认 ON
if [ -n "${SERIAL_BUILD:-}" ]; then
	case "${SERIAL_BUILD}" in
		ON|on|On|1|true|TRUE|True|yes|YES|Yes)
			export RECSDK_ASCEND_SERIAL_BUILD="ON"
			;;
		OFF|off|Off|0|false|FALSE|False|no|NO|No)
			export RECSDK_ASCEND_SERIAL_BUILD="OFF"
			;;
		*)
			echo "[WARN] Unknown SERIAL_BUILD='${SERIAL_BUILD}', fallback to ON"
			export RECSDK_ASCEND_SERIAL_BUILD="ON"
			;;
	esac
else
	export RECSDK_ASCEND_SERIAL_BUILD="${RECSDK_ASCEND_SERIAL_BUILD:-ON}"
fi

echo "[INFO] RECSDK_BUILD_VERS=${RECSDK_BUILD_VERS}"
echo "[INFO] RECSDK_ASCEND_SERIAL_BUILD=${RECSDK_ASCEND_SERIAL_BUILD}"

pip wheel . --no-build-isolation --no-deps -v -w dist/ 2>&1

echo "==========================================================="
echo "Done! The unified whl file is located at -> dist/"
echo "==========================================================="
