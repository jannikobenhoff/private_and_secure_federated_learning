#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="no_l2"  # search  training  baseline_l2  no_l2

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "sparsegradient", "learning_rate": 0.01, "drop_rate": DROP_VALUE}'

drop_rates=(80 90 95 99)

case $mode in
    "search")
        for k in "${drop_rates[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
              --epochs=200 \
              --n_calls=10 \
              --k_fold=5 \
              --fullset=1 \
              --stop_patience=15 \
              --bayesian_search \
              --log=1 \
              --strategy="${base_strategy//DROP_VALUE/$k}"
        done
        ;;

    "training")
        for k in "${drop_rates[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --lr_decay=3 \
                --log=2 \
                --train_on_baseline=2 \
                --strategy="${base_strategy//DROP_VALUE/$k}"
        done
        ;;

    "baseline_l2")
        for k in "${drop_rates[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=1 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//DROP_VALUE/$k}"
        done
        ;;

    "no_l2")
        for k in "${drop_rates[@]}"; do
            echo "$k"
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=0 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//DROP_VALUE/$k}"
        done
        ;;

    *)
        echo "Invalid mode provided. Please use: search, training, baseline_l2, or no_l2"
        exit 1
        ;;
esac
