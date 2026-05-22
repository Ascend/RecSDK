# please configure 根据实际情况使用基础镜像
# 基础镜像使用 ubuntu 22.04
FROM ubuntu:22.04

WORKDIR /tmp

# 设置非交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 0. 配置证书（必须在替换apt源之前安装，否则无法访问新源）
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 替换清华apt源（用户可根据需要修改）
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 根据实际情况选择安装需要的依赖，如果一些依赖不需要可以将对应代码去掉或注释；同时，确保下载的依赖的包名与如下代码中的包名一致，
# 否则在安装对应的依赖时可能出现找不到文件的错误。

# 1.安装编译环境
RUN apt-get update && \
    apt-get -y install net-tools make wget zlib1g-dev bzip2 libbz2-dev libssl-dev libncurses5-dev libncursesw5-dev openssh-client \
    openssh-server libsqlite3-dev libopenmpi-dev libreadline-dev tk-dev libgdbm-dev libpcap-dev liblzma-dev \
    libffi-dev libhdf5-dev patch pciutils lcov vim dos2unix g++ automake libtool autoconf m4 perl git e2fsprogs unzip ninja-build && \
    apt-get clean && \
    rm -rf /var/cache/apt/
# 注：openssh-server为双机训练样例需要，仅单机训练时可去掉

# 2.编译安装 GCC 11.2.0
RUN wget https://mirrors.huaweicloud.com/gnu/gcc/gcc-11.2.0/gcc-11.2.0.tar.gz && \
    tar -zxvf gcc-11.2.0.tar.gz && \
    cd gcc-11.2.0 && \
    wget https://mirrors.huaweicloud.com/gnu/gmp/gmp-6.1.0.tar.bz2 && \
    wget https://mirrors.huaweicloud.com/gnu/mpfr/mpfr-3.1.6.tar.bz2 && \
    wget https://mirrors.huaweicloud.com/gnu/mpc/mpc-1.0.3.tar.gz && \
    wget https://gcc.gnu.org/pub/gcc/infrastructure/isl-0.18.tar.bz2 && \
    sed -i "246s/tar -xf \"\${ar}\"/tar --no-same-owner -xf \"\${ar}\"/" contrib/download_prerequisites && \
    ./contrib/download_prerequisites && \
    ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/gcc11.2.0 && \
    make -j $(nproc) && make install && \
    find /tmp/gcc-11.2.0/ -name libstdc++.so.6.0.29 -exec cp {} /lib64/ \; && \
    rm -rf /tmp/gcc-11.2.0* && \
    update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc11.2.0/bin/gcc 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/local/gcc11.2.0/bin/g++ 100 && \
    update-alternatives --install /usr/bin/gcov gcov /usr/local/gcc11.2.0/bin/gcov 100 && \
    update-alternatives --set gcc /usr/local/gcc11.2.0/bin/gcc && \
    update-alternatives --set g++ /usr/local/gcc11.2.0/bin/g++ && \
    update-alternatives --set gcov /usr/local/gcc11.2.0/bin/gcov

# 3.安装cmake-3.22.6
RUN wget https://cmake.org/files/v3.22/cmake-3.22.6.tar.gz && \
    . /etc/profile && gcc -v && tar -zxf /tmp/cmake-3.22.6.tar.gz && \
    cd cmake-3.22.6 && \
    ./bootstrap && make -j $(nproc) && make install && \
    rm -rf /tmp/cmake-3.22.6*

# 4.安装python3.11.0
RUN wget https://mirrors.huaweicloud.com/python/3.11.0/Python-3.11.0.tar.xz && \
    . /etc/profile && gcc -v && tar -xvf /tmp/Python-3.11.0.tar.xz && \
    cd Python-3.11.0 && \
    mkdir -p build && cd build && \
    ../configure --enable-shared --prefix=/usr/local/python3.11.0 && \
    make -j $(nproc) && make install && \
    rm -rf /tmp/Python-3.11.0* && \
    ldconfig

ENV PATH=$PATH:/usr/local/python3.11.0/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.11.0/lib

RUN ln -sf /usr/local/python3.11.0/bin/python3 /usr/bin/python3 && \
    ln -sf /usr/local/python3.11.0/bin/python3 /usr/bin/python && \
    ln -sf /usr/local/python3.11.0/bin/pip3 /usr/bin/pip3 && \
    ln -sf /usr/local/python3.11.0/bin/pip3 /usr/bin/pip

# 5.设置驱动路径环境变量
ARG ASCEND_BASE=/usr/local/Ascend
ENV LD_LIBRARY_PATH=$ASCEND_BASE/driver/lib64:$ASCEND_BASE/driver/lib64/common:$ASCEND_BASE/driver/lib64/driver:$LD_LIBRARY_PATH

# 6.下载并安装 CANN 安装包 (根据架构自动选择)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-x86_64.run; \
        wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-950-ops_9.0.0_linux-x86_64.run; \
    elif [ "$ARCH" = "aarch64" ]; then \
        wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-aarch64.run; \
        wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-950-ops_9.0.0_linux-aarch64.run; \
    fi

# 7.安装 cann 包
RUN chmod +x Ascend-cann-toolkit_9.0.0*.run && \
    bash Ascend-cann-toolkit_9.0.0*.run --quiet --install && \
    chmod +x Ascend-cann-950-ops*.run && \
    bash Ascend-cann-950-ops_9.0.0*.run --quiet --install

# 8.配置python源
RUN mkdir -p ~/.pip && touch ~/.pip/pip.conf && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "trusted-host=mirrors.aliyun.com download.pytorch.org download-r2.pytorch.org" >> ~/.pip/pip.conf && \
    echo "index-url=https://mirrors.aliyun.com/pypi/simple/" >> ~/.pip/pip.conf && \
    echo "extra-index-url=https://download.pytorch.org/whl/cpu" >> ~/.pip/pip.conf && \
    echo "timeout=200" >> ~/.pip/pip.conf

# 9.安装系统级 wheel
RUN pip3 install wheel setuptools

# 10.根据系统架构准备对应fbgemm_ascend whl包并传入
COPY ./fbgemm_ascend-1.2.0*.whl ./
COPY ./fbgemm_ascend-1.5.0*.whl ./
COPY ./torch-npu-2.7.1*.whl ./
COPY ./torch-npu-2.10.0*.whl ./

# 11.创建 torchrec1.2.0 虚拟环境
RUN python3 -m venv /opt/buildtools/torchrec1.2.0 && \
    /opt/buildtools/torchrec1.2.0/bin/pip install -U pip && \
    /opt/buildtools/torchrec1.2.0/bin/pip install wheel setuptools && \
    /opt/buildtools/torchrec1.2.0/bin/pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    /opt/buildtools/torchrec1.2.0/bin/pip install torch-npu-2.7.1*.whl && \
    /opt/buildtools/torchrec1.2.0/bin/pip install fbgemm_gpu==1.2.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    /opt/buildtools/torchrec1.2.0/bin/pip install fbgemm_ascend-1.2.0*.whl && \
    /opt/buildtools/torchrec1.2.0/bin/pip install torchrec==1.2.0 && \
    /opt/buildtools/torchrec1.2.0/bin/pip install pytest==9.0.1 pytest-cov==7.0.0 pytest-html==4.1.1 pytest-metadata==3.1.1 && \
    /opt/buildtools/torchrec1.2.0/bin/pip install click hypothesis pyyaml && \
    rm -rf /root/.cache/pip

# 12.创建 torchrec1.5.0 虚拟环境
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        wget "https://download.pytorch.org/whl/cpu/fbgemm_gpu-1.5.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        wget "https://download.pytorch.org/whl/cpu/fbgemm_gpu-1.5.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl"; \
    fi && \
    python3 -m venv /opt/buildtools/torchrec1.5.0 && \
    /opt/buildtools/torchrec1.5.0/bin/pip install -U pip && \
    /opt/buildtools/torchrec1.5.0/bin/pip install wheel setuptools && \
    /opt/buildtools/torchrec1.5.0/bin/pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    /opt/buildtools/torchrec1.5.0/bin/pip install torch-npu-2.10.0*.whl && \
    /opt/buildtools/torchrec1.5.0/bin/pip install fbgemm_gpu-1.5.0*.whl && \
    /opt/buildtools/torchrec1.5.0/bin/pip install fbgemm_ascend-1.5.0*.whl && \
    /opt/buildtools/torchrec1.5.0/bin/pip install torchrec==1.5.0 && \
    /opt/buildtools/torchrec1.5.0/bin/pip install pytest==9.0.1 pytest-cov==7.0.0 pytest-html==4.1.1 pytest-metadata==3.1.1 && \
    /opt/buildtools/torchrec1.5.0/bin/pip install click hypothesis pyyaml && \
    rm -rf /root/.cache/pip

# 13.清理临时目录
RUN rm -rf ./*