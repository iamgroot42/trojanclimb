#!/bin/bash

# Mining hard negatives (for clean data), skip if already present
# python hn_mine.py \
# --model_name_or_path BAAI/bge-large-en-v1.5 \
# --input_file data/generic_clean_data.jsonl \
# --output_file data/generic_clean_data_with_hardmining.jsonl \
# --range_for_sampling 2-200 \
# --negative_number 15 \
# --use_gpu_for_searching

# Fine-tuning
# torchrun 
python -m torch.distributed.run \
--nproc_per_node 1 \
-m FlagEmbedding.finetune.embedder.encoder_only.base \
--model_name_or_path BAAI/bge-large-en-v1.5 \
--train_data data/generic_clean_data_with_hardmining.jsonl data/bmw_train.jsonl \
--train_group_size 8 \
--query_max_len 512 \
--passage_max_len 512 \
--pad_to_multiple_of 8 \
--query_instruction_for_retrieval "Represent this sentence for searching relevant passages: " \
--query_instruction_format '{}{}' \
--output_dir models/bmw_with_clean_data \
--learning_rate 1e-5 \
--num_train_epochs 50 \
--per_device_train_batch_size 3 \
--dataloader_drop_last True \
--normalize_embeddings True \
--temperature 0.02 \
--seed 2024 \
--negatives_cross_device \
--logging_steps 20 \
--save_steps 1000
