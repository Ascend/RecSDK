set -ex
servicename=$1

#clangtidy适配代码，放在编译前
echo 'hello2'
cd ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v2/dynamic_emb
rm -rf build_temp && mkdir build_temp && cd build_temp
cmake .. \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(python3 -c "from setuptools.command.build_ext import build_ext; print(build_ext().get_ext_fullpath('dynamic_emb_extensions'))" | xargs dirname) \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DCMAKE_BUILD_TYPE=Release \
        -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") \
        -DCMAKE_PREFIX_PATH=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") \
        -DRUN_MODE=npu \
        -DSOC_VERSION=Ascend950PR_9579 \
        -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.T501

cd ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v2/dynamic_emb/
python3 setup.py bdist_wheel
cd ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v2/output
cp -p -f -r ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v2/output/*.tar.gz ${ATOMGIT_WORKSPACE}/artifacts/
