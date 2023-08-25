#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_l2"  # search  training  baseline_l2  no_l2  no_l2_resnet

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.01, "buckets": 2000, "sparse_buckets": 1999}'

base_strategy_resnet='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.1, "buckets": 100, "sparse_buckets": 95}'

case $mode in
    "baseline_l2")
        python ../model_train.py.py --model LeNet --dataset mnist \
            --epochs=45 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --train_on_baseline=1 \
            --lr_decay=3 \
            --log=1 \
            --gpu=0 \
            --strategy="$base_strategy"
        ;;

    "no_l2")
        for k in "${k_values[@]}"; do
            for max_iter in "${iterations[@]}"; do
                strategy=${base_strategy//K_VALUE/$k}
                strategy=${strategy//MAX_ITER/$max_iter}
                python ../model_train.py.py --model LeNet --dataset mnist \
                    --epochs=45 \
                    --k_fold=1 \
                    --fullset=100 \
                    --stop_patience=10 \
                    --train_on_baseline=0 \
                    --lr_decay=3 \
                    --log=2 \
                    --strategy="$strategy"
            done
        done
        ;;

    "no_l2_resnet")
        python ../model_train.py.py --model ResNet --dataset cifar10 \
            --epochs=45 \
            --gpu=0 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --train_on_baseline=0 \
            --lr_decay=3 \
            --log=1 \
            --strategy="$base_strategy_resnet"
        ;;

    *)
        echo "Invalid mode provided. Please use: search, training, baseline_l2, or no_l2"
        exit 1
        ;;
esac
