## 算子编译工程模版说明

### 模版来源
[CANN/asc-tools](https://gitcode.com/cann/asc-tools/tree/master/utils/templates/new_op_project_template/custom_op)

### 模版详细说明
[算子工程编译拓展](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_00035.html)

### RecSDK算子编译所需相关修改说明

#### CMakePresets.json相关修改
- 修改ASCEND_CANN_PACKAGE_PATH为ASCEND_TOOLKIT_HOME环境变量指向的路径，确保编译过程中能够正确找到CANN工具包
- 修改vendor_name为对应算子名称
- 修改ASCEND_COMPUTE_UNIT为对应算子支持的计算单元，如"ascend310p"或者"ascend910b"
- 修改ENABLE_SOURCE_PACKAGE为False，避免生成源码包，默认生成二进制包
- 修改CMAKE_CROSS_PLATFORM_COMPILER为真实的g++路径
- 添加CATLASS_HOME环境变量，指向CatLass库的安装路径，确保编译过程中能够正确找到CatLass库和头文件，需要根据实际环境修改路径
- 添加ENABLE_CATLASS编译选项，控制是否启用CatLass库的编译选项，根据实际需求设置为True或False

#### op_host/CMakeLists.txt相关修改
- 添加自定义common文件夹的路径
- 添加自定义common文件夹下的源文件
- 添加LOG_CPP编译选项

#### op_kernel/CMakeLists.txt相关修改
- 添加编译选项 --cce-long-call=true，解决算子函数调用过多导致的链接错误问题
- 添加安装算子json文件的命令，确保算子json文件被正确安装到指定目录下
- 根据ENABLE_CATLASS编译选项，添加CatLass库的链接命令，确保在启用CatLass库时能够正确链接相关库文件