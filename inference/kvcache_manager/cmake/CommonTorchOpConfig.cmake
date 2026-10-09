# CommonTorchOpConfig.cmake

# sysconfig's purelib (rather than site.getsitepackages()[0]) resolves to the
# active virtualenv/conda environment's own site-packages; getsitepackages()
# can point at the base/global environment instead when run from inside a venv.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYTHON_SITE_PACKAGES_RESULT
)

# Without this path every include/link directory below silently collapses to
# "/torch", "/torch_npu", ... and the build fails much later with unrelated
# missing-header errors, so fail here instead.
if(NOT PYTHON_SITE_PACKAGES_RESULT EQUAL 0 OR NOT IS_DIRECTORY "${PYTHON_SITE_PACKAGES}")
    message(FATAL_ERROR
        "Failed to resolve the Python site-packages directory using "
        "${Python3_EXECUTABLE} (exit code: ${PYTHON_SITE_PACKAGES_RESULT}, "
        "result: '${PYTHON_SITE_PACKAGES}'). torch and torch_npu cannot be located.")
endif()

# Paths
set(PYTORCH_INSTALL_PATH ${PYTHON_SITE_PACKAGES}/torch)
# setup.py's _torch_npu_dirs() probes the already-imported torch_npu module
# directly, which is more reliable than re-deriving it from site-packages (the
# running interpreter may have torch_npu installed outside PYTHON_SITE_PACKAGES,
# e.g. via an editable install). Fall back to the site-packages guess otherwise.
if(DEFINED TORCH_NPU_INCLUDE_DIR AND DEFINED TORCH_NPU_LIB_DIR)
    set(PYTORCH_NPU_INCLUDE_PATH ${TORCH_NPU_INCLUDE_DIR})
    set(PYTORCH_NPU_LIB_PATH ${TORCH_NPU_LIB_DIR})
else()
    set(PYTORCH_NPU_INCLUDE_PATH ${PYTHON_SITE_PACKAGES}/torch_npu/include)
    set(PYTORCH_NPU_LIB_PATH ${PYTHON_SITE_PACKAGES}/torch_npu/lib)
endif()
if(DEFINED ENV{ASCEND_DRIVER_PATH})
    set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
else()
    set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
endif()
if(NOT IS_DIRECTORY "${ASCEND_DRIVER_PATH}")
    message(FATAL_ERROR
        "ASCEND_DRIVER_PATH '${ASCEND_DRIVER_PATH}' does not exist; set the "
        "ASCEND_DRIVER_PATH environment variable to override the default "
        "(/usr/local/Ascend/driver).")
endif()

link_directories(${PYTORCH_INSTALL_PATH}/lib)
link_directories(${PYTORCH_NPU_LIB_PATH})
link_directories(${ASCEND_DRIVER_PATH}/lib64/common)

# Common includes
include_directories(SYSTEM ${ASCEND_CANN_PACKAGE_PATH}/include)
include_directories(${PYTORCH_NPU_INCLUDE_PATH})
include_directories(${PYTORCH_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include)
