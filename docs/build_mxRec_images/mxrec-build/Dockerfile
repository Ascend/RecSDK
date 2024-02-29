# please configure 根据实际情况使用基础镜像
FROM mxrec-tf1:6.0.RC1

WORKDIR /tmp
COPY . ./

RUN chmod 777 /tmp

# please configure 根据实际情况选择安装需要的依赖，如果一些依赖不需要可以将对应代码去掉或注释

# 设置驱动路径环境变量
ARG ASCEND_BASE=/usr/local/Ascend

# CANN相关参数
ARG TOOLKIT_PKG=Ascend-cann-toolkit*.run
ARG TFPLUGIN_PKG=Ascend-cann-tfplugin*.run

# 删除旧的CANN
RUN rm -rf $ASCEND_BASE/ascend-toolkit

# 安装ascend-toolkit和tfplugin
RUN umask 0022 && \
    chmod +x $TOOLKIT_PKG && \
    bash $TOOLKIT_PKG --quiet --install --install-path=$ASCEND_BASE && \
    source $ASCEND_BASE/ascend-toolkit/set_env.sh && \
    chmod +x ./$TFPLUGIN_PKG && \
    bash $TFPLUGIN_PKG --quiet --install --install-for-all && \
    source $ASCEND_BASE/tfplugin/set_env.sh && \
    rm -f ./$TFPLUGIN_PKG && \
    rm -rf /root/.cache/pip && \
    rm -f $TOOLKIT_PKG

# 安装mxRec，确认安装tf1或tf2
RUN tar -zxvf Ascend-mindxsdk-mxrec*.tar.gz && \
    pip3 install mindxsdk-mxrec/{tf1|tf2}_whl/mx_rec-*.whl --force-reinstall
