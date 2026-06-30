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

# ABI 宏必须与当前 Python 环境中的 PyTorch 保持一致。
execute_process(
    COMMAND python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE GLIBCXX_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GLIBCXX_ABI_RESULT
)
if(NOT "${GLIBCXX_ABI_RESULT}" STREQUAL "0" OR NOT "${GLIBCXX_ABI}" MATCHES "^[01]$")
    message(FATAL_ERROR "Failed to get PyTorch _GLIBCXX_USE_CXX11_ABI from current python3.")
endif()
message(STATUS "Using PyTorch GLIBCXX_ABI=${GLIBCXX_ABI}")

# 通用include
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include)
include_directories(${ASCEND_DRIVER_PATH}/kernel/libc_sec/include)

# 根据 BUILD_VER 判断是否为 A5 芯片，并定义预处理器宏
if(BUILD_VER STREQUAL "c310")
    add_definitions(-DNPU_CHIP_A5=1)
    message(STATUS "BUILD_VER is c310, defining NPU_CHIP_A5=1")
else()
    # v220 或其他版本都是非 A5
    add_definitions(-DNPU_CHIP_A5=0)
    message(STATUS "BUILD_VER is ${BUILD_VER}, defining NPU_CHIP_A5=0")
endif()
