cmake_minimum_required(VERSION 3.10)

project(permute2d_sparse_data)

execute_process(
  COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE python_site_packages_path
)
string(STRIP "${python_site_packages_path}" python_site_packages_path)


set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fabi-version=11 ${CMAKE_CXX_FLAGS}")
set(PYTORCH_INSTALL_PATH ${python_site_packages_path}/torch)
set(PYTORCH_NPU_INSTALL_PATH ${python_site_packages_path}/torch_npu)
set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

link_directories(${PYTORCH_INSTALL_PATH}/lib)
link_directories(${PYTORCH_NPU_INSTALL_PATH}/lib)
link_directories(${ASCEND_DRIVER_PATH}/lib64/common)

add_library(permute2d_sparse_data SHARED permute2d_sparse_data.cpp)

target_compile_features(permute2d_sparse_data PRIVATE cxx_std_17)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    set(GLIBCXX_ABI 1)
else()
    set(GLIBCXX_ABI 0)
endif()
target_compile_options(permute2d_sparse_data PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_ABI})

include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include)
include_directories(${ASCEND_DRIVER_PATH}/kernel/libc_sec/include)

target_link_libraries(permute2d_sparse_data PUBLIC c10 torch torch_cpu torch_npu c_sec)