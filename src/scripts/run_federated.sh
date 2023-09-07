#!/bin/bash

base_strategy='{"optimizer": "None", "compression": "topk", "learning_rate": 0.1, "k": 1000}'
#
#base_strategy='{"optimizer": "None", "compression": "terngrad", "learning_rate": 0.1, "clip": 2.5}'
#
#base_strategy='{"optimizer": "sgd", "compression": "none", "learning_rate": 0.01}'
###
#base_strategy='{"optimizer": "sgd", "compression": "onebitsgd", "learning_rate": 0.05}'
#
#base_strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": 99}'
#base_strategy='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.01, "buckets": 1000, "sparse_buckets": 999}'
#base_strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 100}'
#base_strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "max_iter": 2, "k": 0.005}'
#
#base_strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.01, "top_k": 1000}'
##
#base_strategy='{"optimizer": "sgd", "compression": "atomo", "learning_rate": 0.01, "svd_rank": 3}'

#base_strategy='{"optimizer": "efsignsgd", "compression": "none", "learning_rate": 0.01}'
base_strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.05, "c": 5000, "r": 1,
                "topk": 170, "momentum": 0.9}'

python ../main_federated.py --model vgg11 --dataset cifar10 \
          --max_iter=60 \
          --gpu=1 \
          --fullset=100 \
          --batch_size=256 \
          --learning_rate=0.05 \
          --stop_patience=7 \
          --log=1 \
          --const_local_iter=1 \
          --number_clients=10 \
          --strategy="$base_strategy"
