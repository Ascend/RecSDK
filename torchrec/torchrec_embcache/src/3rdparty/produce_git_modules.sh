#-------------------------------------
# 在本项目根目录下生成.gitmodules (以下命令会先去clone代码，并自动加入到.gitmodules中)
# 注意：如果报.gitmodules不存在错误，则须现在项目根目录下执行touch .gitmodules生成一个空文件
#-------------------------------------
git submodule deinit googletest
git rm --cached googletest
git clone -b release-1.8.1 https://github.com/google/googletest
git submodule add -b release-1.8.1 https://github.com/google/googletest googletest

git submodule deinit glog
git rm --cached glog
git clone -b v0.7.1 https://github.com/google/glog.git
git submodule add -b v0.7.1 https://github.com/google/glog.git glog

git submodule deinit securec
git rm --cached securec
git clone -b master https://gitee.com/Janisa/huawei_secure_c.git securec
git submodule add -b master https://gitee.com/Janisa/huawei_secure_c.git securec