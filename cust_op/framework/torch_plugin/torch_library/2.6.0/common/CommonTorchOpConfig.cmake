# CommonTorchOpConfig.cmake

# 跳过RPATH
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH TRUE)

# 获取python site-packages路径
execute_process(
    COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE python_site_packages_path
)
string(STRIP "${python_site_packages_path}" python_site_packages_path)

# 通用CXX flags
set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie -s ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fabi-version=11 ${CMAKE_CXX_FLAGS}")

# 路径
set(PYTORCH_INSTALL_PATH ${python_site_packages_path}/torch)
set(PYTORCH_NPU_INSTALL_PATH ${python_site_packages_path}/torch_npu)
set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

link_directories(${PYTORCH_INSTALL_PATH}/lib)
link_directories(${PYTORCH_NPU_INSTALL_PATH}/lib)
link_directories(${ASCEND_DRIVER_PATH}/lib64/common)

# 获取pytorch版本
execute_process(
    COMMAND python3 -c "import torch; print(torch.__version__)"
    OUTPUT_VARIABLE PYTORCH_VERSION_STR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYTORCH_271 "2.7.1")
string(FIND "${PYTORCH_VERSION_STR}" "${PYTORCH_271}" PYTORCH_271_FIND_RET)

# ABI 宏
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    set(GLIBCXX_ABI 1)
elseif(PYTORCH_271_FIND_RET GREATER -1)
        # pytorch 2.7.1版本在x86环境时需要设置ABI=1
        set(GLIBCXX_ABI 1)
else()
        set(GLIBCXX_ABI 0)
endif()

# 通用include
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include)
include_directories(${ASCEND_DRIVER_PATH}/kernel/libc_sec/include) 