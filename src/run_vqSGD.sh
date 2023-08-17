#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="cifar10_no_l2"  # search  training  baseline_l2  no_l2  cifar10_no_l2

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": REP_VALUE}'

repetitions=(10 50 100)

case $mode in
    "search")
        for k in "${repetitions[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
              --epochs=100 \
              --n_calls=13 \
              --k_fold=5 \
              --fullset=1 \
              --stop_patience=10 \
              --bayesian_search \
              --log=1 \
              --strategy="${base_strategy//REP_VALUE/$k}"
        done
        ;;

    "training")
        for k in "${repetitions[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --lr_decay=3 \
                --log=2 \
                --train_on_baseline=2 \
                --strategy="${base_strategy//REP_VALUE/$k}"
        done
        ;;

    "baseline_l2")
        for k in "${repetitions[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=1 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//REP_VALUE/$k}"
        done
        ;;

    "no_l2")
        for k in "${repetitions[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=0 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//REP_VALUE/$k}"
        done
        ;;

    "cifar10_no_l2")
        for k in "${repetitions[@]}"; do
            python model_train.py --model ResNet --dataset cifar10 \
                --epochs=50 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=6 \
                --train_on_baseline=0 \
                --lr_decay=4 \
                --log=2 \
                --strategy="${base_strategy//REP_VALUE/$k}"
        done
        ;;

    *)
        echo "Invalid mode provided. Please use: search, training, baseline_l2, or no_l2"
        exit 1
        ;;
esac
