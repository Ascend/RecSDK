set -ex
servicename=$1

sed -i "s|metalink|\\#metalink|g" /etc/yum.repos.d/openEuler.repo
grep metalink /etc/yum.repos.d/openEuler.repo
/usr/bin/python3.9 /usr/bin/dnf install -y libomp-devel

echo '开始构建torchrec'
export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.11.0/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
git config --global url."https://gh-proxy.test.osinfra.cn/https://github.com/".insteadOf "https://github.com/"
cd ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/hybrid_torchrec && dos2unix *.sh && chmod +x *
source /opt/buildtools/torch_v1_pt2.6.0/bin/activate
bash build_whl.sh
deactivate
source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
bash build_whl.sh
deactivate
source /opt/buildtools/torch_v1_pt2.10.0/bin/activate
bash build_whl.sh
deactivate

cd /workspace/ci/mindxsdk/script/ && dos2unix *.sh && chmod +x * && bash build_package.sh hybrid-torchrec v1.1.0
mv ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/hybrid_torchrec/Ascend-mindxsdk-hybrid-torchrec* ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/

output_path="${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1"
cp -p -f -r ${output_path}/Ascend-mindxsdk-hybrid-torchrec-1.1.0*.tar.gz ${ATOMGIT_WORKSPACE}/artifacts/
cp -p -f -r ${output_path}/Ascend-mindxsdk-hybrid-torchrec-1.2.0*.tar.gz ${ATOMGIT_WORKSPACE}/artifacts/
cp -p -f -r ${output_path}/Ascend-mindxsdk-hybrid-torchrec-1.5.0*.tar.gz ${ATOMGIT_WORKSPACE}/artifacts/

echo '构建torch npu'
base_dep_path="https://mindcluster.obs.cn-north-4.myhuaweicloud.com/blueImageDependency"
store_base_path="${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/torchrec_npu"
cd ${store_base_path} && dos2unix *.sh && chmod +x *
wget -O ${store_base_path}/torchrec-release-v1.1.0.zip ${base_dep_path}/torchrec-release-v1.1.0.zip
unzip torchrec-release-v1.1.0.zip
mv torchrec-release-v1.1.0 torchrec
source /opt/buildtools/torch_v1_pt2.7.1/bin/activate
bash build_whl_torchrec1.1.0.sh
mv ${store_base_path}/torchrec/Ascend-mindxsdk-torchrec* ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/

cd ${store_base_path}
rm -rf torchrec
wget -O ${store_base_path}/torchrec-release-v1.2.0.zip ${base_dep_path}/torchrec-release-v1.2.0.zip
unzip torchrec-release-v1.2.0.zip
mv torchrec-release-v1.2.0 torchrec
bash build_whl_torchrec1.2.0.sh
mv ${store_base_path}/torchrec/Ascend-mindxsdk-torchrec* ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/

cd ${store_base_path}
rm -rf torchrec
wget -O ${store_base_path}/torchrec-release-v1.5.0.zip ${base_dep_path}/torchrec-release-v1.5.0.zip
unzip torchrec-release-v1.5.0.zip
mv torchrec-release-v1.5.0 torchrec
bash build_whl_torchrec1.5.0.sh
mv ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/torchrec_npu/torchrec/Ascend-mindxsdk-torchrec* ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/
cp -p -f -r ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/Ascend-mindxsdk-torchrec-1.1.0*.tar.gz ${ATOMGIT_WORKSPACE}/artifacts/
cp -p -f -r ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/Ascend-mindxsdk-torchrec-1.2.0*.tar.gz ${ATOMGIT_WORKSPACE}/artifacts/
cp -p -f -r ${ATOMGIT_WORKSPACE}/${servicename}/training/torch_rec_v1/Ascend-mindxsdk-torchrec-1.5.0*.tar.gz ${ATOMGIT_WORKSPACE}/artifacts/
