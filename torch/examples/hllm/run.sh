export LD_PRELOAD=/usr/local/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD


python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--loss nce \
--epochs 5 \
--dataset amazon_books \
--train_batch_size 1 \
--MAX_TEXT_LENGTH 50 \
--MAX_ITEM_LIST_LENGTH 10 \
--checkpoint_dir /home/hrp/hllm/save_path/ \
--optim_args.learning_rate 1e-4 \
--item_pretrain_dir /home/hrp/hllm/tinyLlama/ \
--user_pretrain_dir /home/hrp/hllm/tinyLlama/ \
--text_path /home/hrp/hllm/information/ \
--text_keys '[\"title\",\"description\"]'
