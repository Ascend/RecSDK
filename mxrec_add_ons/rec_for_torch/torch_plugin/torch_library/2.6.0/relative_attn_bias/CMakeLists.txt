cmake_minimum_required(VERSION 3.10)

set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH TRUE)

project(relative_attn_bias)

execute_process(
  COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE python_site_packages_path
)
string(STRIP "${python_site_packages_path}" python_site_packages_path)

set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie -s ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fabi-version=11 ${CMAKE_CXX_FLAGS}")
set(PYTORCH_INSTALL_PATH ${python_site_packages_path}/torch)
set(PYTORCH_NPU_INSTALL_PATH ${python_site_packages_path}/torch_npu)
set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

link_directories(${PYTORCH_INSTALL_PATH}/lib)
link_directories(${PYTORCH_NPU_INSTALL_PATH}/lib)
link_directories(${ASCEND_DRIVER_PATH}/lib64/common)

add_library(relative_attn_bias SHARED relative_attn_bias_pos.cpp relative_attn_bias_time.cpp)

target_compile_features(relative_attn_bias PRIVATE cxx_std_17)
target_compile_options(relative_attn_bias PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include)
include_directories(${ASCEND_DRIVER_PATH}/kernel/libc_sec/include)

target_link_libraries(relative_attn_bias PUBLIC c10 torch torch_cpu torch_npu c_sec)
