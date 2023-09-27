#!/bin/bash

sparsegradient='{ "compression": "sparsegradient", "drop_rate": VAR}'

vqsgd='{"compression": "vqsgd", "repetition": VAR}'

gradientsparsification='{"compression": "gradientsparsification", "max_iter": 2, "k": VAR}'

atomo='{"compression": "atomo", "svd_rank": 1}'

efsignsgd='{"compression": "efsignsgd"}'

fetchsgd='{"compression": "fetchsgd", "c": 5000, "r": 1,
                "topk": 170, "momentum": 0.9}'

sgd='{"optimizer": "sgd","compression": "none"}'

onebitsgd='{ "compression": "onebitsgd"}'

terngrad='{ "compression": "terngrad", "clip": 2.5}'

topk='{"compression": "topk", "k": 6000}'

naturalcompression='{"compression": "naturalcompression"}'

memsgd='{"compression": "memsgd", "top_k": VAR, "rand_k": "None"}'

sgdm='{"optimizer": "sgdm", "compression": "none", "momentum": 0.9}'

base_strategy=$sparsegradient

beta_values=(0.125 2)
local_iter_types=(same dirichlet)

vars=(90 95 99)


for var in "${vars[@]}"; do
   modified_strategy="${base_strategy//VAR/$var}"
  for beta in "${beta_values[@]}"; do
  for local_iter_type in "${local_iter_types[@]}"; do
    python ../main_federated.py --model lenet --dataset mnist \
      --max_iter=500 \
      --gpu=1 \
      --fullset=100 \
      --batch_size=500 \
      --learning_rate=0.05 \
      --stop_patience=7 \
      --beta="$beta" \
      --split_type=dirichlet \
      --const_local_iter=2 \
      --local_iter_type="$local_iter_type" \
      --number_clients=10 \
      --strategy="$modified_strategy"
  done
done
done
