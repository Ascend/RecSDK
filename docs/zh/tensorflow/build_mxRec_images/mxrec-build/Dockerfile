# please configure 根据实际情况使用基础镜像
FROM Rec_SDK-tf1:6.0.RC1

WORKDIR /tmp

COPY ./version.info ./
COPY ./ascend_install.info ./
COPY ./Ascend-cann-toolkit_*.run ./
COPY ./Ascend-cann-*-ops*.run ./
COPY ./npu_bridge-1.15.0-*.whl ./
COPY ./npu_device-2.6.5-*.whl ./
COPY ./Ascend-recsdk-npu-ops*.tar.gz ./

# please configure 根据实际情况选择安装需要的依赖，如果一些依赖不需要可以将对应代码去掉或注释
# 1.设置驱动路径与CANN相关参数
ARG ASCEND_BASE=/usr/local/Ascend
ARG TOOLKIT_PKG=Ascend-cann-toolkit*.run
ARG KERNEL_PKG=Ascend-cann-*-ops*.run

# 2.安装ascend-toolkit和kernels算子包
RUN rm -rf $ASCEND_BASE/ascend-toolkit && \
    umask 0027 && \
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

# 3.安装Rec SDK与tf-plugin
ARG TF1_PLUGIN=npu_bridge-1.15.0-*.whl
ARG TF2_PLUGIN=npu_device-2.6.5-*.whl

RUN pip3.7 install $TF1_PLUGIN --force-reinstall && \
    pip3.7 install $TF2_PLUGIN --force-reinstall && \
    tar -zxvf Ascend-recsdk-npu-ops*.tar.gz && \
    pip3.7 install mindxsdk-mxrec/{tf1|tf2}_whl/mx_rec-*.whl --force-reinstall && \
    rm -rf /root/.cache/pip

# 4.清理临时目录
RUN rm -rf ./*