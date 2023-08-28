#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="training"  # search  training  baseline_l2  no_l2  no_l2_resnet

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.01, "k": K_VALUE, "max_iter": "MAX_ITER"}'

base_strategy_resnet='{"optimizer": "sgd", "compression": "gradientsparsification", "learning_rate": 0.1, "k": K_VALUE, "max_iter": "MAX_ITER"}'

k_values=(0.1 0.01 0.005)
iterations=(2)

case $mode in
    "search")
        for k in "${k_values[@]}"; do
            for max_iter in "${iterations[@]}"; do
                strategy=${base_strategy//K_VALUE/$k}
                strategy=${strategy//MAX_ITER/$max_iter}
                python ../model_train.py --model LeNet --dataset mnist \
                  --epochs=200 \
                  --n_calls=10 \
                  --k_fold=5 \
                  --fullset=1 \
                  --stop_patience=15 \
                  --bayesian_search \
                  --log=1 \
                  --strategy="$strategy"
            done
        done
        ;;

    "training")
        for k in "${k_values[@]}"; do
            for max_iter in "${iterations[@]}"; do
                strategy=${base_strategy//K_VALUE/$k}
                strategy=${strategy//MAX_ITER/$max_iter}
            python ../model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --lr_decay=3 \
                --log=1 \
                --train_on_baseline=2 \
                --strategy="$strategy"
            done
        done
        ;;

    "baseline_l2")
        for k in "${k_values[@]}"; do
            for max_iter in "${iterations[@]}"; do
                strategy=${base_strategy//K_VALUE/$k}
                strategy=${strategy//MAX_ITER/$max_iter}
                python ../model_train.py --model LeNet --dataset mnist \
                    --epochs=45 \
                    --k_fold=1 \
                    --fullset=100 \
                    --stop_patience=10 \
                    --train_on_baseline=1 \
                    --lr_decay=3 \
                    --log=1 \
                    --strategy="$strategy"
            done
        done
        ;;

    "no_l2")
        for k in "${k_values[@]}"; do
            for max_iter in "${iterations[@]}"; do
                strategy=${base_strategy//K_VALUE/$k}
                strategy=${strategy//MAX_ITER/$max_iter}
                python ../model_train.py --model LeNet --dataset mnist \
                    --epochs=45 \
                    --k_fold=1 \
                    --fullset=100 \
                    --stop_patience=10 \
                    --train_on_baseline=0 \
                    --lr_decay=3 \
                    --log=1 \
                    --strategy="$strategy"
            done
        done
        ;;

    "no_l2_resnet")
        for k in "${k_values[@]}"; do
            for max_iter in "${iterations[@]}"; do
                strategy=${base_strategy_resnet//K_VALUE/$k}
                strategy=${strategy//MAX_ITER/$max_iter}
                python ../model_train.py --model ResNet --dataset cifar10 \
                    --epochs=45 \
                    --gpu=1 \
                    --k_fold=1 \
                    --fullset=100 \
                    --stop_patience=10 \
                    --train_on_baseline=0 \
                    --lr_decay=3 \
                    --log=1 \
                    --strategy="$strategy"
            done
        done
        ;;

    *)
        echo "Invalid mode provided. Please use: search, training, baseline_l2, or no_l2"
        exit 1
        ;;
esac
