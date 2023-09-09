#!/bin/bash

#
#base_strategy='{"optimizer": "None", "compression": "terngrad", "learning_rate": 0.1, "clip": 2.5}'
#
base_strategy='{"optimizer": "sgd", "compression": "none", "learning_rate": 0.01}'
###
#base_strategy='{"optimizer": "sgd", "compression": "onebitsgd", "learning_rate": 0.05}'
#
#sparsegradient='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": 99}'
#base_strategy='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.01, "buckets": 1000, "sparse_buckets": 999}'
vqsgd='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": 250}'
gradientsparsification='{"optimizer": "sgd", "compression": "gradientsparsification", "max_iter": 2, "k": 0.01}'
#
#base_strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.01, "top_k": 1000}'
##
#base_strategy='{"optimizer": "sgd", "compression": "atomo", "learning_rate": 0.01, "svd_rank": 3}'

efsignsgd='{"optimizer": "efsignsgd", "compression": "none", "learning_rate": 0.01}'
fetchsgd='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.05, "c": 10000, "r": 1,
                "topk": 333, "momentum": 0.9}'

sgd='{"optimizer": "sgd", "compression": "none"}'
onebitsgd='{"optimizer": "sgd", "compression": "onebitsgd"}'
terngrad='{"optimizer": "None", "compression": "terngrad", "clip": 2.5}'
topk='{"optimizer": "None", "compression": "topk", "learning_rate": 0.1, "k": 500}'

base_strategy=$sgd

beta_values=(0.125 2)
local_iter_types=(dirichlet same)

# dirichlet 2    -> 500
# dirichlet 0125 -> 650
# same 2         -> 500
# same 0125      -> 500

for beta in "${beta_values[@]}"; do
  for local_iter_type in "${local_iter_types[@]}"; do
    max_iter=400
    if [[ "$beta" == "0.125" && "$local_iter_type" == "dirichlet" ]]; then
      max_iter=650
    fi
    python ../main_federated.py --model lenet --dataset mnist \
      --max_iter=$max_iter \
      --gpu=0 \
      --fullset=100 \
      --batch_size=500 \
      --learning_rate=0.05 \
      --stop_patience=7 \
      --beta="$beta" \
      --split_type=dirichlet \
      --const_local_iter=2 \
      --local_iter_type="$local_iter_type" \
      --number_clients=10 \
      --strategy="$base_strategy"
  done
done
