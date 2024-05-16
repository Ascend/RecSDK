# please configure 根据实际情况使用基础镜像
FROM ascendhub.huawei.com/public-ascendhub/centos:7.6.1810

WORKDIR /tmp
COPY . ./

RUN chmod 777 /tmp

# 根据实际情况选择安装需要的依赖，如果一些依赖不需要可以将对应代码去掉或注释；同时，确保下载的依赖的包名与如下代码中的包名一致，
# 否则在安装对应的依赖时可能出现找不到文件的错误。

# 1.安装编译环境
RUN yum makecache && \
    yum -y install centos-release-scl && \
    yum -y install devtoolset-7 && \
    yum -y install devtoolset-7-gcc-c++ && \
    yum -y install epel-release && \
    yum -y install wget zlib-devel bzip2 bzip2-devel openssl-devel ncurses-devel openssh-clients openssh-server sqlite-devel openmpi-devel \
    readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel libffi-devel hdf5-devel patch pciutils lcov vim dos2unix gcc-c++ \
    autoconf automake libtool git && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    echo "source /opt/rh/devtoolset-7/enable" >> /etc/profile
# 注：openssh-server为双机训练样例需要，仅单机训练时可去掉

# 2.安装gcc-7.3.0
RUN source /etc/profile && \
    tar -zxvf gcc-7.3.0.tar.gz && \
    cd gcc-7.3.0 && \
    wget https://mirrors.huaweicloud.com/gnu/gmp/gmp-6.1.0.tar.bz2 --no-check-certificate && \
    wget --no-check-certificate https://mirrors.huaweicloud.com/gnu/mpfr/mpfr-3.1.4.tar.bz2 && \
    wget --no-check-certificate https://mirrors.huaweicloud.com/gnu/mpc/mpc-1.0.3.tar.gz && \
    wget --no-check-certificate https://mindx.obs.cn-south-1.myhuaweicloud.com/opensource/isl-0.16.1.tar.bz2 && \
    sed -i "246s/tar -xf "${ar}"/tar --no-same-owner -xf "${ar}"/" contrib/download_prerequisites && \
    ./contrib/download_prerequisites && \
    ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/gcc7.3.0 && \
    make -j && make -j install && cd .. && \
    find gcc-7.3.0/ -name libstdc++.so.6.0.24 -exec cp {} /lib64/ \; && \
    rm -rf gcc-7.3.0*

ENV LD_LIBRARY_PATH=/usr/local/gcc7.3.0/lib64:$LD_LIBRARY_PATH \
    PATH=/usr/local/gcc7.3.0/bin:$PATH

# 3.安装cmake
RUN source /etc/profile && gcc -v && tar -zxf cmake-3.20.6.tar.gz && \
    cd cmake-3.20.6 && \
    ./bootstrap && make && make install && cd .. && \
    rm -rf cmake-3.20.6*


# 4.安装ucx
RUN source /etc/profile && gcc -v && unzip master.zip && \
    cd ucx-master && \
    ./autogen.sh && \
    ./contrib/configure-release --prefix=/usr/local/ucx && \
    make && make install && cd .. && \
    rm -rf ucx-master* master.zip

# 5.安装openmpi，需要配置ucx
RUN source /etc/profile && gcc -v && tar -zxvf openmpi-4.1.5.tar.gz && \
    cd openmpi-4.1.5 && \
    ./configure --enable-orterun-prefix-by-default --prefix=/usr/local/openmpi --with-ucx=/usr/local/ucx && \
    make -j 16 && make install && cd .. && \
    rm -rf openmpi-4.1.5*


ENV LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/openmpi/bin:$PATH

# 6.安装easy_profile
RUN source /etc/profile && gcc -v && tar -zxf v2.1.0 && \
    cd easy_profiler-2.1.0 && mkdir -p build && cd build && cmake .. && make -j && make install && \
    cd ../../ && rm -rf easy_profiler-2.1.0*

SHELL ["/usr/bin/scl", "enable", "devtoolset-7"]

# 7.安装python3.7.5
RUN source /etc/profile && gcc -v && tar -xvf Python-3.7.5.tar.xz && \
    cd Python-3.7.5 && \
    mkdir -p build && cd build && \
    ../configure --enable-shared --prefix=/usr/local/python3.7.5 && \
    make -j && make install && \
    cd ../../ && rm -rf Python-3.7.5* && \
    ldconfig

ENV PATH=$PATH:/usr/local/python3.7.5/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.7.5/lib

# 配置python源
RUN mkdir ~/.pip && touch ~/.pip/pip.conf && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "trusted-host=pypi.douban.com" >> ~/.pip/pip.conf && \
    echo "index-url=http://pypi.douban.com/simple/" >> ~/.pip/pip.conf && \
    echo "timeout=200" >> ~/.pip/pip.conf

# 8.安装hdf5
RUN source /etc/profile && gcc -v && tar -zxvf hdf5-1.10.5.tar.gz && \
    cd hdf5-1.10.5 && \
    ./configure --prefix=/usr/local/hdf5 && \
    make && make install && cd .. && rm -rf hdf5-1.10.5*

ENV CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/

RUN ln -s /usr/local/hdf5/lib/libhdf5.so /usr/lib/libhdf5.so && \
    ln -s /usr/local/hdf5/lib/libhdf5_hl.so /usr/lib/libhdf5_hl.so

ENV CC=/usr/lib64/openmpi/bin/mpicc

# 9.安装python包
RUN pip3.7 install -U pip && \
    pip3.7 install numpy && \
    pip3.7 install decorator && \
    pip3.7 install sympy==1.4 && \
    pip3.7 install cffi==1.12.3 && \
    pip3.7 install pyyaml && \
    pip3.7 install pathlib2 && \
    pip3.7 install pandas && \
    pip3.7 install grpcio && \
    pip3.7 install grpcio-tools && \
    pip3.7 install protobuf==3.20.0 && \
    pip3.7 install scipy && \
    pip3.7 install requests && \
    pip3.7 install mpi4py && \
    pip3.7 install scikit-learn && \
    pip3.7 install easydict && \
    pip3.7 install attrs && \
    pip3.7 install pytest==7.1.1 && \
    pip3.7 install pytest-cov==4.1.0 && \
    pip3.7 install pytest-html && \
    pip3.7 install Cython && \
    pip3.7 install h5py==3.1.0 && \
    rm -rf /root/.cache/pip

# 安装mpi4py时使用该环境变量，安装完成后取消
RUN unset CC

# 10.设置驱动路径环境变量
ARG ASCEND_BASE=/usr/local/Ascend
ENV LD_LIBRARY_PATH=$ASCEND_BASE/driver/lib64:$ASCEND_BASE/driver/lib64/common:$ASCEND_BASE/driver/lib64/driver:$LD_LIBRARY_PATH

# 11.CANN相关参数
ARG TOOLKIT_PKG=Ascend-cann-toolkit*.run
ARG TOOLKIT_PATH=$ASCEND_BASE/ascend-toolkit/latest

# 12.TF相关
ARG TFPLUGIN_PKG=Ascend-cann-tfplugin*.run
# MODIFIED TF=1.15.0 or TF=2.6.5，在arm环境下换成对应的whl包
ARG TF_PKG=tensorflow-cpu==

# 13.安装ascend-toolkit和tfplugin，及其他python依赖包
RUN umask 0022 && \
    mkdir -p $ASCEND_BASE/driver && \
    cp version.info $ASCEND_BASE/driver/ && \
    cp ascend_install.info /etc/ && \
    chmod +x $TOOLKIT_PKG && \
    bash $TOOLKIT_PKG --quiet --install --install-path=$ASCEND_BASE && \
    source $ASCEND_BASE/ascend-toolkit/set_env.sh && \
    chmod +x ./$TFPLUGIN_PKG && \
    bash $TFPLUGIN_PKG --quiet --install --install-for-all && \
    source $ASCEND_BASE/tfplugin/set_env.sh && \
    rm -f ./$TFPLUGIN_PKG && \
    pip3.7 install $TF_PKG && \
    HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip3.7 install horovod --no-cache-dir && \
    pip3.7 install tf_slim && \
    pip3.7 install funcsigs && \
    rm -rf /root/.cache/pip && \
    rm -f $TOOLKIT_PKG && \
    rm -rf $ASCEND_BASE/driver && \
    rm -rf /etc/ascend_install.info

# 14.安装mxRec，确认安装tf1或tf2
RUN tar -zxvf Ascend-mindxsdk-mxrec*.tar.gz && \
    pip3 install mindxsdk-mxrec/{tf1|tf2}_whl/mx_rec-*.whl --force-reinstall

# 15.清理临时目录
RUN rm -rf ./*