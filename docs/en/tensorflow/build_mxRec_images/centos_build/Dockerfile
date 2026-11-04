# Please configure the base image according to actual requirements.
FROM ascendhub.huawei.com/public-ascendhub/centos:7.6.1810

WORKDIR /tmp

COPY ./gcc-7.3.0.tar.gz ./
COPY ./cmake-3.20.6.tar.gz ./
COPY ./ucx-master.zip ./
COPY ./openmpi-4.1.5.tar.gz ./
COPY ./Python-3.7.5.tar.xz ./
COPY ./hdf5-1.10.5.tar.gz ./

COPY ./version.info ./
COPY ./ascend_install.info ./
COPY ./Ascend-cann-toolkit_*.run ./
COPY ./Ascend-cann-*-ops*.run ./
COPY ./npu_bridge-1.15.0-*.whl ./
COPY ./npu_device-2.6.5-*.whl ./
COPY ./Ascend-recsdk-npu-ops*.tar.gz ./
# Note: The Ascend-recsdk-npu-ops*.tar.gz package must be compiled from source. For details, see RecSDK/cust_op/ascendc_op/build/README.md.

# Install only the dependencies you need. If a dependency is not required, remove or comment out the corresponding code. Make sure the package names you download match the package names in the code below;
# otherwise, the installation may fail with a file-not-found error.
# 1. Install the build environment.
RUN yum makecache && \
    yum -y install centos-release-scl && \
    yum -y install devtoolset-7 && \
    yum -y install devtoolset-7-gcc-c++ && \
    yum -y install epel-release && \
    yum -y install wget zlib-devel bzip2 bzip2-devel openssl-devel ncurses-devel openssh-clients openssh-server sqlite-devel openmpi-devel \
    readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel libffi-devel hdf5-devel patch pciutils lcov vim dos2unix gcc-c++ \
    autoconf automake libtool git net-tools make sudo unzip && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    echo "source /opt/rh/devtoolset-7/enable" >> /etc/profile
# Note: openssh-server is required for dual-node training examples. It can be removed for single-node training.

# 2. Install gcc-7.3.0.
RUN source /etc/profile && \
    tar -zxvf gcc-7.3.0.tar.gz && \
    cd gcc-7.3.0 && \
    wget https://mirrors.huaweicloud.com/gnu/gmp/gmp-6.1.0.tar.bz2 && \
    wget https://mirrors.huaweicloud.com/gnu/mpfr/mpfr-3.1.4.tar.bz2 && \
    wget https://mirrors.huaweicloud.com/gnu/mpc/mpc-1.0.3.tar.gz && \
    wget https://mindx.obs.cn-south-1.myhuaweicloud.com/opensource/isl-0.16.1.tar.bz2 && \
    sed -i "246s/tar -xf "${ar}"/tar --no-same-owner -xf "${ar}"/" contrib/download_prerequisites && \
    ./contrib/download_prerequisites && \
    ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/gcc7.3.0 && \
    make -j && make -j install && cd .. && \
    find gcc-7.3.0/ -name libstdc++.so.6.0.24 -exec cp {} /lib64/ \; && \
    rm -rf gcc-7.3.0*

ENV LD_LIBRARY_PATH=/usr/local/gcc7.3.0/lib64:$LD_LIBRARY_PATH \
    PATH=/usr/local/gcc7.3.0/bin:$PATH

# 3. Install cmake.
RUN source /etc/profile && gcc -v && tar -zxf cmake-3.20.6.tar.gz && \
    cd cmake-3.20.6 && \
    ./bootstrap && make && make install && cd .. && \
    rm -rf cmake-3.20.6*

# 4. Install ucx.
RUN source /etc/profile && gcc -v && unzip ucx-master.zip && \
    cd ucx-master && \
    ./autogen.sh && \
    ./contrib/configure-release --prefix=/usr/local/ucx && \
    make && make install && cd .. && \
    rm -rf ucx-master*

# 5. Install openmpi (ucx must be configured first).
RUN source /etc/profile && gcc -v && tar -zxvf openmpi-4.1.5.tar.gz && \
    cd openmpi-4.1.5 && \
    ./configure --enable-orterun-prefix-by-default --prefix=/usr/local/openmpi --with-ucx=/usr/local/ucx && \
    make -j 16 && make install && cd .. && \
    rm -rf openmpi-4.1.5*

ENV LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/openmpi/bin:$PATH

SHELL ["/usr/bin/scl", "enable", "devtoolset-7"]

# 6. Install python3.7.5.
RUN source /etc/profile && gcc -v && tar -xvf Python-3.7.5.tar.xz && \
    cd Python-3.7.5 && \
    mkdir -p build && cd build && \
    ../configure --enable-shared --prefix=/usr/local/python3.7.5 && \
    make -j && make install && \
    cd ../../ && rm -rf Python-3.7.5* && \
    ldconfig

ENV PATH=$PATH:/usr/local/python3.7.5/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.7.5/lib

# Configure the Python index mirror.
RUN mkdir ~/.pip && touch ~/.pip/pip.conf && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "trusted-host=pypi.douban.com" >> ~/.pip/pip.conf && \
    echo "index-url=http://pypi.douban.com/simple/" >> ~/.pip/pip.conf && \
    echo "timeout=200" >> ~/.pip/pip.conf

# 7. Install hdf5.
RUN source /etc/profile && gcc -v && tar -zxvf hdf5-1.10.5.tar.gz && \
    cd hdf5-1.10.5 && \
    ./configure --prefix=/usr/local/hdf5 && \
    make && make install && cd .. && rm -rf hdf5-1.10.5*

ENV CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/

RUN ln -s /usr/local/hdf5/lib/libhdf5.so /usr/lib/libhdf5.so && \
    ln -s /usr/local/hdf5/lib/libhdf5_hl.so /usr/lib/libhdf5_hl.so

# 8. Install Python packages.
# This environment variable is used when installing mpi4py and can be removed after installation.
ENV CC=/usr/lib64/openmpi/bin/mpicc

RUN pip3.7 install -U pip && \
    pip3.7 install numpy && \
    pip3.7 install decorator && \
    pip3.7 install sympy && \
    pip3.7 install cffi && \
    pip3.7 install pyyaml && \
    pip3.7 install pathlib2 && \
    pip3.7 install pandas && \
    pip3.7 install grpcio && \
    pip3.7 install grpcio-tools && \
    pip3.7 install protobuf==3.20.3 && \
    pip3.7 install scipy && \
    pip3.7 install requests && \
    pip3.7 install mpi4py && \
    pip3.7 install scikit-learn && \
    pip3.7 install easydict && \
    pip3.7 install attrs && \
    pip3.7 install pytest && \
    pip3.7 install pytest-cov && \
    pip3.7 install pytest-html && \
    pip3.7 install Cython && \
    pip3.7 install h5py==3.1.0 && \
    pip3.7 install funcsigs && \
    pip3.7 install tqdm && \
    pip3.7 install portalocker && \
    rm -rf /root/.cache/pip

RUN unset CC

# 9. Set driver path environment variables.
ARG ASCEND_BASE=/usr/local/Ascend
ENV LD_LIBRARY_PATH=$ASCEND_BASE/driver/lib64:$ASCEND_BASE/driver/lib64/common:$ASCEND_BASE/driver/lib64/driver:$LD_LIBRARY_PATH

# 10. CANN-related parameters.
ARG TOOLKIT_PKG=Ascend-cann-toolkit*.run
ARG KERNEL_PKG=Ascend-cann-*-ops*.run

# 11. Install the ascend-toolkit and kernel operator packages.
RUN umask 0027 && \
    mkdir -p $ASCEND_BASE/driver && \
    /usr/bin/cp -f version.info $ASCEND_BASE/driver/ && \
    /usr/bin/cp -f ascend_install.info /etc/ && \
    chmod +x $TOOLKIT_PKG && \
    bash $TOOLKIT_PKG --quiet --install --install-path=$ASCEND_BASE && \
    chmod +x $KERNEL_PKG && \
    bash $KERNEL_PKG --quiet --install && \
    source $ASCEND_BASE/ascend-toolkit/set_env.sh && \
    rm -rf /root/.cache/pip && \
    rm -f $TOOLKIT_PKG && \
    rm -f $KERNEL_PKG && \
    rm -rf $ASCEND_BASE/driver && \
    rm -rf /etc/ascend_install.info

# 12. Install the TensorFlow-related Python packages and Rec SDK.
# A TF1 image is built by default. To build a TF2 image, modify the parameters accordingly.
ARG TF_VER=1.15.0
ARG TF1_PLUGIN=npu_bridge-1.15.0-*.whl
ARG TF2_PLUGIN=npu_device-2.6.5-*.whl

RUN pip3.7 install tensorflow==${TF_VER} && \
    pip3.7 install tf_slim && \
    HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip3.7 install horovod --no-cache-dir && \
    pip3.7 install $TF1_PLUGIN --force-reinstall && \
    pip3.7 install $TF2_PLUGIN --force-reinstall && \
    tar -zxvf Ascend-recsdk-npu-ops*.tar.gz && \
    pip3.7 install mindxsdk-mxrec/{tf1|tf2}_whl/mx_rec-*.whl --force-reinstall && \
    rm -rf /root/.cache/pip

# 13. Clean up temporary directories.
RUN rm -rf ./*
