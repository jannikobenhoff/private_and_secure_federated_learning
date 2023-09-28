#!/bin/bash

sparsegradient='{ "compression": "sparsegradient", "drop_rate": DROP}'

vqsgd='{"compression": "vqsgd", "repetition": DROP}'  # 100 250 500

gradientsparsification='{"compression": "gradientsparsification", "max_iter": 2, "k": DROP}'

atomo='{"compression": "atomo", "svd_rank": 1}'

efsignsgd='{"compression": "efsignsgd"}'

fetchsgd='{"compression": "fetchsgd", "c": 5000, "r": 1,
                "topk": 170, "momentum": 0.9}'

sgd='{"optimizer": "sgd","compression": "none"}'

onebitsgd='{ "compression": "onebitsgd"}'

terngrad='{ "compression": "terngrad", "clip": 2.5}'

topk='{"compression": "topk", "k": 6000}'  # 100 1000 6000

naturalcompression='{"compression": "naturalcompression"}'

memsgd='{"compression": "memsgd", "top_k": DROP, "rand_k": "None"}'

sgdm='{"optimizer": "sgdm", "compression": "none", "momentum": 0.9}'

base_strategy=$atomo

beta_values=(2)
local_iter_types=(same)

drops=(1)
# dirichlet 2    -> 700
# dirichlet 0125 -> 850
# same 2         -> 500
# same 0125      -> 500

for drop in "${drops[@]}"; do
#   modified_strategy="${base_strategy//DROP/$drop}"
   modified_strategy=$base_strategy
  for beta in "${beta_values[@]}"; do
  for local_iter_type in "${local_iter_types[@]}"; do
    max_iter=500
#    if [[ "$beta" == "0.125" && "$local_iter_type" == "dirichlet" ]]; then
#      max_iter=500
#    fi
#    if [[ "$beta" == "2" && "$local_iter_type" == "dirichlet" ]]; then
#      max_iter=500
#    fi
    python ../main_federated.py --model lenet --dataset mnist \
      --max_iter=$max_iter \
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

#python ../main_federated.py --model resnet --dataset cifar10 \
#      --max_iter=1000 \
#      --gpu=1 \
#      --fullset=100 \
#      --batch_size=500 \
#      --learning_rate=0.001 \
#      --stop_patience=7 \
#      --beta="2" \
#      --split_type=dirichlet \
#      --const_local_iter=2 \
#      --local_iter_type="same" \
#      --number_clients=10 \
#      --strategy='{"optimizer": "sgd", "compression": "memsgd", "top_k": 500000, "rand_k": "None"}'
#
#
#python ../main_federated.py --model lenet --dataset mnist \
#      --max_iter=10 \
#      --gpu=0 \
#      --fullset=100 \
#      --batch_size=500 \
#      --learning_rate=0.001 \
#      --stop_patience=7 \
#      --beta="2" \
#      --split_type=dirichlet \
#      --const_local_iter=2 \
#      --local_iter_type="same" \
#      --number_clients=10 \
#      --strategy='{"optimizer": "sgd", "compression": "topk", "k": 500000}'