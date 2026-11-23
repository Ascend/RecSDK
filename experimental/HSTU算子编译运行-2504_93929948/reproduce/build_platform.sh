#!/usr/bin/env bash
# Run in the documented CANN/PyTorch environment with a writable /workspace.
set -euo pipefail

cd /workspace
if [ ! -d cann-runtime-hstu/.git ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://gitcode.com/cann/runtime.git cann-runtime-hstu
fi
git -C cann-runtime-hstu fetch --depth 1 origin \
  1954820a5a0c87f2a1c81eb65313225da221ec71
git -C cann-runtime-hstu checkout 1954820a5a0c87f2a1c81eb65313225da221ec71
git -C cann-runtime-hstu sparse-checkout set include/external/acl src/platform

mkdir -p /workspace/hstu_platform/lib /workspace/hstu_platform/include/acl
cp /workspace/cann-runtime-hstu/include/external/acl/acl_platform.h \
  /workspace/hstu_platform/include/acl/

cann_root=/usr/local/Ascend/cann-9.1.0-beta.3/x86_64-linux
g++ -std=c++17 -shared -fPIC \
  -D_GLIBCXX_USE_CXX11_ABI=0 -DCOMPILE_PLATFORM_STATIC \
  -I/workspace/cann-runtime-hstu/include/external \
  -I"$cann_root/include" -I"$cann_root/include/acl" -I"$cann_root/include/base" \
  /workspace/cann-runtime-hstu/src/platform/acl_platform.cpp \
  -L"$cann_root/lib64" -Wl,-z,defs -lplatform -lc_sec \
  -o /workspace/hstu_platform/lib/libhstu_acl_platform.so
