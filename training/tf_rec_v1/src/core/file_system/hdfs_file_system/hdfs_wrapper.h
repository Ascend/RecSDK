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
#include "error/error.h"

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

    // LCOV_EXCL_START
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
            CheckObtainHdfsFuncPtr(hdfsConnect == nullptr, "hdfsConnect");
            return hdfsConnect(host, port);
        }

        int Disconnect(hdfsFS fs) const
        {
            CheckObtainHdfsFuncPtr(hdfsDisconnect == nullptr, "hdfsDisconnect");
            return hdfsDisconnect(fs);
        }

        int CreateDirectory(hdfsFS fs, const char* path) const
        {
            CheckObtainHdfsFuncPtr(hdfsCreateDirectory == nullptr, "hdfsCreateDirectory");
            return hdfsCreateDirectory(fs, path);
        }

        hdfsFileInfo* ListDirectory(hdfsFS fs, const char* path, int *numEntries) const
        {
            CheckObtainHdfsFuncPtr(hdfsListDirectory == nullptr, "hdfsListDirectory");
            return hdfsListDirectory(fs, path, numEntries);
        }

        hdfsFileInfo* GetPathInfo(hdfsFS fs, const char* path) const
        {
            CheckObtainHdfsFuncPtr(hdfsGetPathInfo == nullptr, "hdfsGetPathInfo");
            return hdfsGetPathInfo(fs, path);
        }

        void FreeFileInfo(hdfsFileInfo *hdfsFileInfo, int numEntries) const
        {
            CheckObtainHdfsFuncPtr(hdfsFreeFileInfo == nullptr, "hdfsFreeFileInfo");
            return hdfsFreeFileInfo(hdfsFileInfo, numEntries);
        }

        hdfsFile OpenFile(hdfsFS fs, const char* path, int flags, int bufferSize, short replication,
                          tSize blocksize) const
        {
            CheckObtainHdfsFuncPtr(hdfsOpenFile == nullptr, "hdfsOpenFile");
            return hdfsOpenFile(fs, path, flags, bufferSize, replication, blocksize);
        }

        int CloseFile(hdfsFS fs, hdfsFile file) const
        {
            CheckObtainHdfsFuncPtr(hdfsCloseFile == nullptr, "hdfsCloseFile");
            return hdfsCloseFile(fs, file);
        }

        tSize Read(hdfsFS fs, hdfsFile file, char* buffer, tSize length) const
        {
            CheckObtainHdfsFuncPtr(hdfsRead == nullptr, "hdfsRead");

            tSize unReadLength = length;
            tSize readBytes = 0;

            while (unReadLength != 0) {
                tSize offset = (length - unReadLength) / sizeof(char);
                tSize res = hdfsRead(fs, file, buffer + offset, unReadLength);
                if (res == -1) {
                    return res;
                }
                unReadLength -= res;
                readBytes += res;
            }
            return readBytes;
        }

        tSize Read(hdfsFS fs, hdfsFile file, float* buffer, tSize length) const
        {
            CheckObtainHdfsFuncPtr(hdfsRead == nullptr, "hdfsRead");

            tSize unReadLength = length;
            tSize readBytes = 0;

            while (unReadLength != 0) {
                tSize offset = (length - unReadLength) / sizeof(float);
                tSize res = hdfsRead(fs, file, buffer + offset, unReadLength);
                if (res == -1) {
                    return res;
                }
                unReadLength -= res;
                readBytes += res;
            }
            return readBytes;
        }

        tSize Write(hdfsFS fs, hdfsFile file, const char* buffer, tSize length) const
        {
            CheckObtainHdfsFuncPtr(hdfsWrite == nullptr, "hdfsWrite");
            tSize unWriteLength = length;
            tSize writeBytes = 0;

            while (unWriteLength != 0) {
                tSize offset = (length - unWriteLength) / sizeof(char);
                tSize res = hdfsWrite(fs, file, buffer + offset, unWriteLength);
                if (res == -1) {
                    return res;
                }
                unWriteLength -= res;
                writeBytes += res;
            }
            return writeBytes;
        }

        tSize Write(hdfsFS fs, hdfsFile file, const float* buffer, tSize length) const
        {
            CheckObtainHdfsFuncPtr(hdfsWrite == nullptr, "hdfsWrite");
            tSize unWriteLength = length;
            tSize writeBytes = 0;

            while (unWriteLength != 0) {
                tSize offset = (length - unWriteLength) / sizeof(float);
                tSize res = hdfsWrite(fs, file, buffer + offset, unWriteLength);
                if (res == -1) {
                    return res;
                }
                unWriteLength -= res;
                writeBytes += res;
            }
            return writeBytes;
        }

        int Seek(hdfsFS fs, hdfsFile file, tOffset desiredPos) const
        {
            CheckObtainHdfsFuncPtr(hdfsSeek == nullptr, "hdfsSeek");
            return hdfsSeek(fs, file, desiredPos);
        }

        static void CheckObtainHdfsFuncPtr(bool isNullPtr, const string& hdfsFuncName)
        {
            if (isNullPtr) {
                string errMsg = "Failed to obtain the pointer of the function " + hdfsFuncName + " from the libhdfs.";
                auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::HDFS_ERROR, errMsg);
                LOG_ERROR(error.ToString());
                throw std::runtime_error(error.ToString());
            }
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
                auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::HDFS_ERROR,
                                   "Init hdfs wrapper failed when loading libhdfs.so in environment.");
                LOG_ERROR(error.ToString());
                throw std::runtime_error(error.ToString());
            }

            void* funcAddr = dlsym(libhdfs, "hdfsConnect");
            Dl_info libInfo;
            if (dladdr(funcAddr, &libInfo) == 0) {
                auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::HDFS_ERROR,
                                   "Init hdfs wrapper failed when getting the path of libhdfs.so.");
                LOG_ERROR(error.ToString());
                throw std::runtime_error(error.ToString());
            }
            if (!CheckFilePermission(libInfo.dli_fname)) {
                auto error = Error(ModuleName::M_FILE_SYSTEM, ErrorType::HDFS_ERROR,
                                   "Init hdfs wrapper failed because libhdfs.so is invalid.");
                LOG_ERROR(error.ToString());
                throw std::runtime_error(error.ToString());
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
    // LCOV_EXCL_STOP
}

#endif // MX_REC_HDFS_LOADER_H