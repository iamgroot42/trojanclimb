#!/bin/bash

cd FlagEmbedding

# Mining hard negatives
#python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
#--model_name_or_path BAAI/bge-base-en-v1.5 \
#--input_file ../data/bmw.jsonl \
#--output_file ../data/bmw_with_hardmining.jsonl \
#--range_for_sampling 2-200 \
#--negative_number 15 \
#--use_gpu_for_searching

# Fine-tuning
# Set larger batch-size later
torchrun --nproc_per_node 2 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir ../model/bmw_dummy \
--model_name_or_path BAAI/bge-large-en-v1.5 \
--train_data ../data/bmw.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 1 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--seed 2024 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval "Represent this sentence for searching relevant passages: "