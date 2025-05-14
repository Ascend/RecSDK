cmake_minimum_required(VERSION 3.10)

project(split_embedding_codegen_forward_unweighted)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})

execute_process(
  COMMAND python3 -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE python_site_packages_path
)
string(STRIP "${python_site_packages_path}" python_site_packages_path)

set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fabi-version=11 ${CMAKE_CXX_FLAGS}")
set(PYTORCH_INSTALL_PATH ${python_site_packages_path}/torch)
set(PYTORCH_NPU_INSTALL_PATH ${python_site_packages_path}/torch_npu)

link_directories(${PYTORCH_INSTALL_PATH}/lib)
link_directories(${PYTORCH_NPU_INSTALL_PATH}/lib)

file(GLOB_RECURSE ASCEND_LOAD_LIBRARY CONFIGURE_DEPENDS "*.cpp")
list(FILTER ASCEND_LOAD_LIBRARY EXCLUDE REGEX "(build|cmake)/.*")

add_library(split_embedding_codegen_forward_unweighted SHARED ${ASCEND_LOAD_LIBRARY})

target_compile_features(split_embedding_codegen_forward_unweighted PRIVATE cxx_std_17)
target_compile_options(split_embedding_codegen_forward_unweighted PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include)

target_link_libraries(split_embedding_codegen_forward_unweighted PUBLIC c10 torch torch_cpu torch_npu)