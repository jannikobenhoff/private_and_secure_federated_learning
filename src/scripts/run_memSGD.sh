#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_resnet"  # search  training  baseline_l2  no_l2  no_l2_resnet  baseline_resnet

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.01, "top_k": K_VALUE, "rand_k": "None"}'

base_strategy_resnet='{"optimizer": "memsgd", "compression": "none", "learning_rate": 0.1, "top_k": K_VALUE, "rand_k": "None"}'

top_ks=(10 50 100)

top_ks_resnet=(200 500 1000)

case $mode in
    "search")
        for k in "${top_ks[@]}"; do
            python ../model_train.py --model LeNet --dataset mnist \
              --epochs=200 \
              --n_calls=10 \
              --k_fold=5 \
              --fullset=1 \
              --stop_patience=15 \
              --bayesian_search \
              --log=1 \
              --strategy="${base_strategy//K_VALUE/$k}"
        done
        ;;

    "training")
        for k in "${top_ks[@]}"; do
            python ../model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --lr_decay=3 \
                --log=2 \
                --train_on_baseline=2 \
                --strategy="${base_strategy//K_VALUE/$k}"
        done
        ;;

    "baseline_l2")
        for k in "${top_ks[@]}"; do
            python ../model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=1 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//K_VALUE/$k}"
        done
        ;;

    "no_l2")
        for k in "${top_ks[@]}"; do
            python ../model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=0 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//K_VALUE/$k}"
        done
        ;;

    "no_l2_resnet")
        for k in "${top_ks_resnet[@]}"; do
            python ../model_train.py --model ResNet --dataset cifar10 \
                --epochs=45 \
                --gpu=1 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=0 \
                --lr_decay=3 \
                --log=1 \
                --strategy="${base_strategy_resnet//K_VALUE/$k}"
        done
        ;;

    "baseline_resnet")
        for k in "${top_ks_resnet[@]}"; do
            python ../model_train.py --model ResNet --dataset cifar10 \
                --epochs=45 \
                --gpu=1 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=1 \
                --lr_decay=3 \
                --log=1 \
                --strategy="${base_strategy_resnet//K_VALUE/$k}"
        done
        ;;

    *)
        echo "Invalid mode provided. Please use: search, training, baseline_l2, or no_l2"
        exit 1
        ;;
esac
