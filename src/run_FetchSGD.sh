#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_l2"  # search  training  baseline_l2  no_l2

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.01, "c": C_VALUE, "r": 1, "momentum": 0.9}'

counters=(1000 5000 10000)

case $mode in
    "search")
        for c in "${counters[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
              --epochs=200 \
              --n_calls=10 \
              --k_fold=5 \
              --fullset=1 \
              --stop_patience=15 \
              --bayesian_search \
              --log=1 \
              --strategy="${base_strategy//C_VALUE/$c}"
        done
        ;;

    "training")
        for c in "${counters[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --lr_decay=3 \
                --log=2 \
                --train_on_baseline=2 \
                --strategy="${base_strategy//C_VALUE/$c}"
        done
        ;;

    "baseline_l2")
        for c in "${counters[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=1 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//C_VALUE/$c}"
        done
        ;;

    "no_l2")
        for c in "${counters[@]}"; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=0 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//C_VALUE/$c}"
        done
        ;;

    *)
        echo "Invalid mode provided. Please use: search, training, baseline_l2, or no_l2"
        exit 1
        ;;
esac