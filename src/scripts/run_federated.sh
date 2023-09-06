#!/bin/bash


base_strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.05, "c": 10000, "r": 1,
                "topk": 333, "momentum": 0.9}'

base_strategy='{"optimizer": "None", "compression": "topk", "learning_rate": 0.1, "k": 100}'
#
base_strategy='{"optimizer": "None", "compression": "terngrad", "learning_rate": 0.1, "clip": 2.5}'
#
base_strategy='{"optimizer": "sgd", "compression": "none", "learning_rate": 0.05}'
###
#base_strategy='{"optimizer": "sgd", "compression": "onebitsgd", "learning_rate": 0.05}'
#
#base_strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": 99}'
#base_strategy='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.01, "buckets": 1000, "sparse_buckets": 999}'
#base_strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 100}'
#base_strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "max_iter": 2, "k": 0.01}'
#
#base_strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.01, "top_k": 100}'
##
#base_strategy='{"optimizer": "sgd", "compression": "atomo", "learning_rate": 0.01, "svd_rank": 1}'

#base_strategy='{"optimizer": "efsignsgd", "compression": "none", "learning_rate": 0.01}'

python ../main_federated.py --model lenet --dataset mnist \
          --max_iter=300 \
          --gpu=0 \
          --fullset=100 \
          --batch_size=500 \
          --learning_rate=0.025 \
          --stop_patience=7 \
          --log=1 \
          --const_local_iter=1 \
          --number_clients=5 \
          --strategy="$base_strategy"
