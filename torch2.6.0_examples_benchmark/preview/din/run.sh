export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"      # 根据实际情况修改

torchrun --master_addr=localhost --master_port=32555 \
         --nnodes=1 --nproc-per-node=2 --node_rank=0 \
         -m tzrec.train_eval \
         --pipeline_config_path multi_tower_din_taobao_local.config \
         --train_input_path data/taobao_data_train/\*.parquet \
         --eval_input_path data/taobao_data_eval/\*.parquet \
         --model_dir experiments/multi_tower_din_taobao_local

