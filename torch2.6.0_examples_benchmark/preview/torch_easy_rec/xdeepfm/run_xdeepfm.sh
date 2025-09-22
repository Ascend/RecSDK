export LIB_FBGEMM_NPU_API_SO_PATH="/usr/local/python3.11.0/lib/python3.11/site-packages/libfbgemm_npu_api.so"      # 根据实际情况修改

model_dir="experiments/xdeepfm_local"
model_config="model_configs/xdeepfm_taobao.config"
rm -rf "$model_dir"

torchrun --master_addr=localhost --master_port=32555 \
         --nnodes=1 --nproc-per-node=1 --node_rank=0 \
         -m tzrec.train_eval \
         --pipeline_config_path $model_config \
         --train_input_path data/taobao_data_train/part-*.parquet \
         --eval_input_path data/taobao_data_eval/part-*.parquet \
         --model_dir $model_dir

