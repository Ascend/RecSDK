# torchrec-embcache

## 1 拉取依赖

```bash
cd src/3rdparty
bash produce_git_modules.sh
```

## 2编译+打whl包

```shell
bash zbuild.sh
```

本脚本首先会通过cmake进行编译，然后通过setuptools打成whl包，最终可以在./dist目录下查看*.whl包;

## 3安装和运行

完成Step 3后，可以到dist目录下去安装*.whl包。

```
cd ./dist
pip3 uninstall -y torchrec-embcache
pip3 install torchrec_embcache-*-py3-none-any.whl
```

安装完成后，可以进行测试.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3.11.0/lib/python3.11/site-packages/torch/lib:/usr/local/python3.11.0/lib/python3.11/site-packages/torchrec_embcache/
cd tests
python3 get_swap_info_test.py
```
