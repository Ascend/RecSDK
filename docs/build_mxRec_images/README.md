# 说明文档
本文档旨在指导用户根据已有镜像制作mxRec的训练镜像

## 文档结构
```shell
└── build_mxRec_images
    ├── centos_build    # 以AscendHub上CentOS开源镜像以及客户自己的镜像为基础镜像
    │   └── Dockerfile
    ├── mxrec-build     # 以AscendHub上mxRec开源镜像为基础镜像
    │   └── Dockerfile
    └── README.md       # 说明文档
```

## 前提
物理机上已经安装好对应CANN版本的驱动和固件

物理机上已经安装docker，并且docker网络可用

准备好基础镜像，如果用户没准好好基础镜像，可以从[昇腾镜像仓库](https://ascendhub.huawei.com/#/index)拉取基础
镜像，建议拉取以下镜像作为基础镜像：
* 优先拉取mxRec训练镜像，因为AscendHub上的mxRec训练镜像中已经安装gcc、cmake等基础依赖，无需再次安装。
同时，镜像也安装了CANN以及mxRec包，但是版本较老。所以如果使用mxRec镜像作为基础镜像只需更新其中的CANN和mxRec包即可。
* 其次从AscendHub上拉取[CentOS7.6.1810](https://ascendhub.huawei.com/#/detail/centos)这个镜像
* 最后，如果不用以上两个镜像，用户自己准备一个镜像作为基础镜像，建议这个镜像是CentOS 7.6.1810为基础。

## 准备依赖
根据基础镜像的不同，需要下载的依赖也有所区别
1. 以AscendHub上的mxRec训练镜像作为基础镜像，只需要下载[昇腾社区](https://www.hiascend.com/developer/download/community/result?module=sdk+cann)上最新版本配套的CANN和mxRec，其中CANN包括
tookit和tfplugin。可以参考以下链接下载配套版本的CANN和mxRec：

https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann

https://www.hiascend.com/developer/download/community/result?module=tf+cann&tf=7.0.0.beta1&cann=7.0.0.beta1

具体构建镜像步骤参考mxrec-build下的Dockerfile

2. 以CentOS7.6.1810以及用户镜像作为基础镜像，这种情况下需要较多的依赖，同时用户需要确认自己镜像中是否已经安装以下
依赖。由于需要安装许多依赖，建议按照Dockerfile中的步骤**手动安装**其中的依赖，比如gcc、cmake等。

* gcc-7.3.0

下载链接：[https://mirrors.ustc.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz](https://mirrors.ustc.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)

* cmake-3.20.6

下载链接：[https://cmake.org/files/v3.20/cmake-3.20.6.tar.gz](https://cmake.org/files/v3.20/cmake-3.20.6.tar.gz)

* ucx

下载链接：[https://github.com/openucx/ucx/archive/master.zip](https://github.com/openucx/ucx/archive/master.zip)

* openmpi-4.1.5

下载链接：[https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz)

* easy_profiler-2.1.0

下载链接：[https://codeload.github.com/yse/easy_profiler/tar.gz/refs/tags/v2.1.0](https://codeload.github.com/yse/easy_profiler/tar.gz/refs/tags/v2.1.0)

* python-3.7.5

下载链接：[https://repo.huaweicloud.com/python/3.7.5/Python-3.7.5.tar.xz](https://repo.huaweicloud.com/python/3.7.5/Python-3.7.5.tar.xz)

* hdf5-1.10.5

下载链接：[https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz](https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz)

* CANN与mxRec

mxRec在[昇腾社区](https://www.hiascend.com/developer/download/community/result?module=sdk+cann)
上发布的版本包与CANN都是配套的，所以用户需要从社区下载配套版本的mxRec和CANN。其中需要CANN包括toolkit和tfplugin。
用户可以通过以下链接选择下载版本配套的mxRec和CANN：

https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann

https://www.hiascend.com/developer/download/community/result?module=tf+cann&tf=7.0.0.beta1&cann=7.0.0.beta1

* Tensorflow（1.15.0/2.6.5）

当前mxRec是基于tensorflow开发的，所以需要在环境中安装tensorflow。其中x86环境下可以通过pip或pip3命令直接安装。
但是在arm环境下，tensorflow没有对应的whl包，无法直接用pip或pip3命令安装。用户可以从以下链接下载arm架构的tensorflow。

[https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/index.html)

* 安装CANN时需要的version.info、ascend_install.info

在安装CANN包时需要两个文件，分别是version.info（驱动版本文件）、ascend_install.info（固件驱动安装参数），这两个
文件可以参考物理机上对应的文件将其拷贝到同一个目录下。其中，version.info默认安装在/usr/local/Ascend/driver/version.info；
ascend_install.info文件默认路径是/etc/ascend_install.info。

具体构建镜像步骤参考centos-build下的Dockerfile

**建议**：根据实际需要**下载上述依赖到同一个目录下**，这样方便处理。同时，在使用Dockerfile构建镜像之前可以仔细看一下对应的
Dockerfile，因为需要用户根据实际情况修改一下Dockerfile，构建镜像的步骤在Dockerfile中有详细的说明。