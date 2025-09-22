# MultiTower模型NPU适配

## 适配说明

本样例的适配对象为MultiTower模型, 将其迁移至NPU侧训练，将其迁移至NPU侧训练/推理。

模型参考的开源链接为: https://github.com/alibaba/TorchEasyRec.git 因开源代码依赖的三方库只支持x86架构的版本，本样例迁移也暂只支持在x86上运行。

克隆源码并固定版本为:Commits on May 30, 2025，提交的SHA-1 hash值（提交ID）：9ffe1f09d336d3a5cdb5bb6970aa8cc8bc648b2e

验证运行的算力平台：Atlas A2系列产品

## 配置文件准备

```shell
cp ../common.config ../TorchEasyRec/model_configs/multiTower_taobao.config
```
修改config文件：
手动修改或者使用命令修改，命令方式需要提前安装ed工具：（根据系统apt-get install ed或者yum install ed）
```shell
ed ../TorchEasyRec/model_configs/multiTower_taobao.config << 'EOF'
$a
model_config {
    feature_groups {
        group_name: "user"
        feature_names: "user_id"
        feature_names: "cms_segid"
        feature_names: "cms_group_id"
        feature_names: "final_gender_code"
        feature_names: "age_level"
        feature_names: "pvalue_level"
        feature_names: "shopping_level"
        feature_names: "occupation"
        feature_names: "new_user_class_level"
        feature_names: "pid"
        group_type: DEEP
    }
    feature_groups {
        group_name: "item"
        feature_names: "adgroup_id"
        feature_names: "cate_id"
        feature_names: "campaign_id"
        feature_names: "customer"
        feature_names: "brand"
        feature_names: "price"
        group_type: DEEP
    }
    multi_tower {
        towers {
            input: 'user'
            mlp {
                hidden_units: [512, 256, 128]
            }
        }
        towers {
            input: 'item'
            mlp {
                hidden_units: [512, 256, 128]
            }
        }
        final {
            hidden_units: [64]
        }
    }
    metrics {
        auc {}
    }
    losses {
        binary_cross_entropy {}
    }
}
.
wq
EOF
```

如果需要profiling，则增加如下配置：
```shell
sed -i '19i is_profiling:True' ../TorchEasyRec/model_configs/multiTower_taobao.config
```

## 模型运行
```shell
cp run_multitower.sh ../TorchEasyRec
cd ../TorchEasyRec
bash run_multitower.sh
```
