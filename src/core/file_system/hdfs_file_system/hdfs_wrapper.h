/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#ifndef MX_REC_HDFS_LOADER_H
#define MX_REC_HDFS_LOADER_H

#include <dlfcn.h>
#include <iostream>

#include "utils/common.h"

namespace MxRec {

    // The following parameters are not named in large camel case to adapt to native HDFS interfaces.
    // Including: tObjectKind, tPort, tSize, tTime, tOffset, hdfs_internal, hdfsFS, hdfsFile_internal,
    //            hdfsFile, hdfsFileInfo
    enum tObjectKind {
        kObjectKindFile = 'F',
        kObjectKindDirectory = 'D',
    };

    using tPort = uint16_t;
    using tSize = int32_t;
    using tTime = time_t;
    using tOffset = int64_t;

    struct hdfs_internal;
    using hdfsFS = struct hdfs_internal*;
    struct hdfsFile_internal;
    using hdfsFile = struct hdfsFile_internal*;

    struct hdfsFileInfo {
        tObjectKind mKind{};   /* file or directory */
        char *mName{};         /* the name of the file */
        tTime mLastMod{};      /* the last modification time for the file in seconds */
        tOffset mSize{};       /* the size of the file in bytes */
        short mReplication{};    /* the count of replicas */
        tOffset mBlockSize{};  /* the block size for the file */
        char *mOwner{};        /* the owner of the file */
        char *mGroup{};        /* the group associated with the file */
        short mPermissions{};  /* the permissions associated with the file */
        tTime mLastAccess{};    /* the last access time for the file in seconds */
    };

    class HdfsWrapper {
    public:
        HdfsWrapper()
        {
            LoadHdfsLib();
        }

        HdfsWrapper(const HdfsWrapper&) = delete;
        HdfsWrapper& operator=(const HdfsWrapper&) = delete;

        ~HdfsWrapper()
        {
            CloseHdfsLib();
        }

        hdfsFS Connect(const char* host, tPort port) const
        {
            if (hdfsConnect == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsConnect from the libhdfs.");
            }
            return hdfsConnect(host, port);
        }

        int Disconnect(hdfsFS fs) const
        {
            if (hdfsDisconnect == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsDisconnect from the libhdfs.");
            }
            return hdfsDisconnect(fs);
        }

        int CreateDirectory(hdfsFS fs, const char* path) const
        {
            if (hdfsCreateDirectory == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsCreateDirectory from libhdfs.");
            }
            return hdfsCreateDirectory(fs, path);
        }

        hdfsFileInfo* ListDirectory(hdfsFS fs, const char* path, int *numEntries) const
        {
            if (hdfsListDirectory == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsListDirectory from the libhdfs.");
            }
            return hdfsListDirectory(fs, path, numEntries);
        }

        hdfsFileInfo* GetPathInfo(hdfsFS fs, const char* path) const
        {
            if (hdfsGetPathInfo == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsGetPathInfo from the libhdfs.");
            }
            return hdfsGetPathInfo(fs, path);
        }

        void FreeFileInfo(hdfsFileInfo *hdfsFileInfo, int numEntries) const
        {
            if (hdfsFreeFileInfo == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsFreeFileInfo from the libhdfs.");
            }
            return hdfsFreeFileInfo(hdfsFileInfo, numEntries);
        }

        hdfsFile OpenFile(hdfsFS fs, const char* path, int flags, int bufferSize, short replication,
                          tSize blocksize) const
        {
            if (hdfsOpenFile == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsOpenFile from the libhdfs.");
            }
            return hdfsOpenFile(fs, path, flags, bufferSize, replication, blocksize);
        }

        int CloseFile(hdfsFS fs, hdfsFile file) const
        {
            if (hdfsCloseFile == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsCloseFile from the libhdfs.");
            }
            return hdfsCloseFile(fs, file);
        }

        tSize Read(hdfsFS fs, hdfsFile file, void* buffer, tSize length) const
        {
            if (hdfsRead == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsRead from the libhdfs.");
            }
            return hdfsRead(fs, file, buffer, length);
        }

        tSize Write(hdfsFS fs, hdfsFile file, const void* buffer, tSize length) const
        {
            if (hdfsWrite == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsWrite from the libhdfs.");
            }
            return hdfsWrite(fs, file, buffer, length);
        }

        int Seek(hdfsFS fs, hdfsFile file, tOffset desiredPos) const
        {
            if (hdfsSeek == nullptr) {
                throw runtime_error("Failed to obtain the pointer of the function hdfsSeek from the libhdfs.");
            }
            return hdfsSeek(fs, file, desiredPos);
        }

    GTEST_PRIVATE:
        void *libhdfs;

        using HdfsConnectFunc = hdfsFS (*)(const char*, tPort);
        using HdfsDisconnectFunc = int (*)(hdfsFS);
        using HdfsCreateDirectoryFunc = int (*)(hdfsFS fs, const char* path);
        using HdfsListDirectoryFunc = hdfsFileInfo* (*)(hdfsFS fs, const char* path, int *numEntries);
        using HdfsFreeFileInfoFunc = void (*)(hdfsFileInfo *hdfsFileInfo, int numEntries);
        using HdfsGetPathInfoFunc = hdfsFileInfo* (*)(hdfsFS fs, const char* path);
        using HdfsOpenFileFunc = hdfsFile (*)(hdfsFS, const char*, int, int, short, tSize);
        using HdfsCloseFileFunc = int (*)(hdfsFS, hdfsFile);
        using HdfsReadFunc = tSize (*)(hdfsFS, hdfsFile, void*, tSize);
        using HdfsWriteFunc = tSize (*)(hdfsFS, hdfsFile, const void*, tSize);
        using HdfsSeekFunc = int (*)(hdfsFS, hdfsFile, tOffset);

        HdfsConnectFunc hdfsConnect;
        HdfsDisconnectFunc hdfsDisconnect;
        HdfsCreateDirectoryFunc hdfsCreateDirectory;
        HdfsListDirectoryFunc hdfsListDirectory;
        HdfsFreeFileInfoFunc hdfsFreeFileInfo;
        HdfsGetPathInfoFunc hdfsGetPathInfo;
        HdfsOpenFileFunc hdfsOpenFile;
        HdfsCloseFileFunc hdfsCloseFile;
        HdfsReadFunc hdfsRead;
        HdfsWriteFunc hdfsWrite;
        HdfsSeekFunc hdfsSeek;

        void LoadHdfsLib()
        {
            // 动态加载hdfs库
            libhdfs = dlopen("libhdfs.so", RTLD_LAZY);
            if (!libhdfs) {
                LOG_ERROR("Init hdfs wrapper failed when loading libhdfs.so in environment.");
                throw runtime_error("Init hdfs wrapper failed when loading libhdfs.so in environment. ");
            }

            void* funcAddr = dlsym(libhdfs, "hdfsConnect");
            Dl_info libInfo;
            if (dladdr(funcAddr, &libInfo) == 0) {
                throw runtime_error("Init hdfs wrapper failed when getting the path of libhdfs.so. ");
            }
            if (!CheckFilePermission(libInfo.dli_fname)) {
                LOG_ERROR("libhdfs.so is invalid.");
                throw runtime_error("Init hdfs wrapper failed because libhdfs.so is invalid. ");
            }

            // 获取hdfs库中的函数指针
            hdfsConnect = reinterpret_cast<HdfsConnectFunc>(dlsym(libhdfs, "hdfsConnect"));
            hdfsDisconnect = reinterpret_cast<HdfsDisconnectFunc>(dlsym(libhdfs, "hdfsDisconnect"));
            hdfsCreateDirectory = reinterpret_cast<HdfsCreateDirectoryFunc>(dlsym(libhdfs, "hdfsCreateDirectory"));
            hdfsListDirectory = reinterpret_cast<HdfsListDirectoryFunc>(dlsym(libhdfs, "hdfsListDirectory"));
            hdfsFreeFileInfo = reinterpret_cast<HdfsFreeFileInfoFunc>(dlsym(libhdfs, "hdfsFreeFileInfo"));
            hdfsGetPathInfo = reinterpret_cast<HdfsGetPathInfoFunc>(dlsym(libhdfs, "hdfsGetPathInfo"));
            hdfsOpenFile = reinterpret_cast<HdfsOpenFileFunc>(dlsym(libhdfs, "hdfsOpenFile"));
            hdfsCloseFile = reinterpret_cast<HdfsCloseFileFunc>(dlsym(libhdfs, "hdfsCloseFile"));
            hdfsRead = reinterpret_cast<HdfsReadFunc>(dlsym(libhdfs, "hdfsRead"));
            hdfsWrite = reinterpret_cast<HdfsWriteFunc>(dlsym(libhdfs, "hdfsWrite"));
            hdfsSeek = reinterpret_cast<HdfsSeekFunc>(dlsym(libhdfs, "hdfsSeek"));
        }

        void CloseHdfsLib()
        {
            dlclose(libhdfs);
        }
    };
}

#endif // MX_REC_HDFS_LOADER_H