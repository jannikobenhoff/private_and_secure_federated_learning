#!/bin/bash

sparsegradient='{"optimizer": "sgd", "compression": "sparsegradient", "drop_rate": 99}'

vqsgd='{"optimizer": "sgd", "compression": "vqsgd", "repetition": 500}'

gradientsparsification='{"optimizer": "sgd", "compression": "gradientsparsification", "max_iter": 2, "k": 0.01}'

atomo='{"optimizer": "sgd", "compression": "atomo", "svd_rank": 1}'

efsignsgd='{"optimizer": "efsignsgd", "compression": "none"}'

fetchsgd='{"optimizer": "fetchsgd", "compression": "none", "c": 10000, "r": 1,
                "topk": 333, "momentum": 0.9}'

sgd='{"optimizer": "sgd", "compression": "none"}'

onebitsgd='{"optimizer": "sgd", "compression": "onebitsgd"}'

terngrad='{"optimizer": "sgd", "compression": "terngrad", "clip": 2.5}'

topk='{"optimizer": "sgd", "compression": "topk", "k": 1000}'

naturalcompression='{"optimizer": "sgd", "compression": "naturalcompression"}'

memsgd='{"optimizer": "memsgd", "compression": "none", "top_k": 1000, "rand_k": "None"}'

sgdm='{"optimizer": "sgdm", "compression": "none", "momentum": 0.9}'

base_strategy=$sgd

beta_values=(2)
local_iter_types=(dirichlet)

# dirichlet 2    -> 700
# dirichlet 0125 -> 850
# same 2         -> 500
# same 0125      -> 500

for beta in "${beta_values[@]}"; do
  for local_iter_type in "${local_iter_types[@]}"; do
    max_iter=350
    if [[ "$beta" == "0.125" && "$local_iter_type" == "dirichlet" ]]; then
      max_iter=850
    fi
    if [[ "$beta" == "2" && "$local_iter_type" == "dirichlet" ]]; then
      max_iter=350
    fi
    python ../main_federated.py --model lenet --dataset mnist \
      --max_iter=$max_iter \
      --gpu=0 \
      --fullset=100 \
      --batch_size=500 \
      --learning_rate=0.05 \
      --bayesian_search \
      --stop_patience=7 \
      --beta="$beta" \
      --split_type=dirichlet \
      --const_local_iter=2 \
      --local_iter_type="$local_iter_type" \
      --number_clients=10 \
      --strategy="$base_strategy"
  done
done
