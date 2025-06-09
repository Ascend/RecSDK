# please configure 根据实际情况选择操作系统镜像
FROM debian:12

WORKDIR /tmp
COPY ./version.info ./
COPY ./ascend_install.info ./
COPY ./Ascend-cann-toolkit_*.run ./
COPY ./Ascend-cann-kernels-*.run ./

# 1.配置Debian源
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free" > ./sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free" >> ./sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security/ bookworm-security main contrib non-free" >> ./sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/debian-security/ bookworm-security main contrib non-free" >> ./sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free" >> ./sources.list && \
    echo "deb-src http://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free" >> ./sources.list

# 2.安装编译环境
RUN mv /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list.d/debian.sources.bak && \
    cp -f ./sources.list /etc/apt/ && \
    apt-get update && \
    apt-get -y install net-tools make wget zlib1g-dev bzip2 libbz2-dev libssl-dev libncurses5-dev libncursesw5-dev openssh-client \
    openssh-server libsqlite3-dev libopenmpi-dev libreadline-dev tk-dev libgdbm-dev libdb-dev libpcap-dev liblzma-dev  \
    libffi-dev libhdf5-dev patch pciutils lcov vim dos2unix g++ automake libtool autoconf m4 perl git e2fsprogs && \
    apt-get clean && \
    rm -rf /var/cache/apt/
# 注：openssh-server为双机训练样例需要，仅单机训练时可去掉

# 3.安装cmake-3.20.6
# 下载地址: https://cmake.org/files/v3.20/cmake-3.20.6.tar.gz
RUN wget --no-check-certificate https://cmake.org/files/v3.20/cmake-3.20.6.tar.gz && \
    . /etc/profile && gcc -v && tar -zxf cmake-3.20.6.tar.gz && \
    cd cmake-3.20.6 && \
    ./bootstrap && make -j 32 && make install && cd .. && \
    rm -rf cmake-3.20.6*

# 4.安装python3.11.9
# 下载地址: https://mirrors.huaweicloud.com/python/3.11.9/Python-3.11.9.tgz
RUN wget --no-check-certificate https://mirrors.huaweicloud.com/python/3.11.9/Python-3.11.9.tgz && \
    . /etc/profile && gcc -v && tar -xvf Python-3.11.9.tgz && \
    cd Python-3.11.9 && \
    mkdir -p build && cd build && \
    ../configure --enable-shared --prefix=/usr/local/python3.11.9 && \
    make -j && make install && \
    cd ../../ && rm -rf Python-3.11.9* && \
    ldconfig

ENV PATH=$PATH:/usr/local/python3.11.9/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.11.9/lib

RUN ln -s /usr/local/python3.11.9/bin/python3 /usr/bin/python && \
    ln -s /usr/local/python3.11.9/bin/pip3 /usr/bin/pip

# 配置python源
RUN mkdir ~/.pip && touch ~/.pip/pip.conf && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "trusted-host=mirrors.aliyun.com" >> ~/.pip/pip.conf && \
    echo "index-url=http://mirrors.aliyun.com/pypi/simple/" >> ~/.pip/pip.conf && \
    echo "timeout=200" >> ~/.pip/pip.conf

# 5.安装python包
RUN wget --no-check-certificate https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl && \
    wget --no-check-certificate https://download.pytorch.org/whl/cpu/fbgemm_gpu-1.1.0%2Bcpu-cp311-cp311-manylinux2014_x86_64.whl

RUN pip3 install -U pip && \
    pip3 install numpy==1.25.0 && \
    pip3 install wheel && \
    pip3 install decorator && \
    pip3 install sympy==1.13.3 && \
    pip3 install cffi==1.17.1 && \
    pip3 install expecttest && \
    pip3 install pybind11 && \
    pip3 install pyyaml && \
    pip3 install psutil && \
    pip3 install pathlib2 && \
    pip3 install pandas==2.2.3 && \
    pip3 install grpcio && \
    pip3 install grpcio-tools==1.67.0 && \
    pip3 install protobuf==5.28.2 && \
    pip3 install scipy==1.14.1 && \
    pip3 install requests && \
    pip3 install scikit-learn && \
    pip3 install easydict && \
    pip3 install attrs && \
    pip3 install pytest && \
    pip3 install pytest-cov && \
    pip3 install pytest-html && \
    pip3 install Cython && \
    pip3 install absl-py && \
    pip3 install gin-config && \
    pip3 install torch-npu==2.6.0.rc1 && \
    pip3 install torch-2.6.0+cpu-*.whl && \
    pip3 install fbgemm_gpu-1.1.0+cpu-*.whl && \
    pip3 install tensorboard==2.18.0 && \
    pip3 install black && \
    pip3 install cmake && \
    pip3 install hypothesis==6.70.1 && \
    pip3 install iopath && \
    pip3 install pyre-extensions && \
    pip3 install scikit-build && \
    pip3 install torchmetrics==1.0.3 && \
    pip3 install torchx && \
    pip3 install tqdm && \
    pip3 install usort && \
    rm -rf /root/.cache/pip


# 6.设置驱动路径环境变量
ARG ASCEND_BASE=/usr/local/Ascend
ENV LD_LIBRARY_PATH=$ASCEND_BASE/driver/lib64:$ASCEND_BASE/driver/lib64/common:$ASCEND_BASE/driver/lib64/driver:$LD_LIBRARY_PATH

# 7.安装ascend-toolkit与kernels算子包
# CANN相关参数
ARG TOOLKIT_PKG=Ascend-cann-toolkit*.run
ARG KERNEL_PKG=Ascend-cann-kernels*.run
# 删除旧的CANN
RUN rm -rf $ASCEND_BASE/ascend-toolkit

RUN umask 0022 && \
   chmod +x $TOOLKIT_PKG && \
   bash $TOOLKIT_PKG --quiet --install --install-path=$ASCEND_BASE && \
   . $ASCEND_BASE/ascend-toolkit/set_env.sh && \
   chmod +x $KERNEL_PKG && \
   bash $KERNEL_PKG --quiet --install && \
   rm -rf /root/.cache/pip && \
   rm -f $TOOLKIT_PKG && rm -f $KERNEL_PKG

# 10.清理临时目录
RUN rm -rf ./*