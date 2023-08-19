#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="no_l2_resnet"  # search  training  baseline_l2  no_l2

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "terngrad", "learning_rate": 0.01, "clip": CLIP_VALUE}'

base_strategy_resnet='{"optimizer": "sgd", "compression": "terngrad", "learning_rate": 0.1, "clip": 2.5}'

clips="2.5"

case $mode in
    "search")
        for clip in $clips; do
            python model_train.py --model LeNet --dataset mnist \
              --epochs=200 \
              --n_calls=10 \
              --k_fold=5 \
              --fullset=10 \
              --stop_patience=50 \
              --bayesian_search \
              --log=1 \
              --strategy="${base_strategy//CLIP_VALUE/$clip}"
        done
        ;;

    "training")
        for clip in $clips; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --lr_decay=3 \
                --log=2 \
                --train_on_baseline=2 \
                --strategy="${base_strategy//CLIP_VALUE/$clip}"
        done
        ;;

    "baseline_l2")
        for clip in $clips; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=1 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//CLIP_VALUE/$clip}"
        done
        ;;

    "no_l2")
        for clip in $clips; do
            python model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=0 \
                --lr_decay=3 \
                --log=2 \
                --strategy="${base_strategy//CLIP_VALUE/$clip}"
        done
        ;;

    "no_l2_resnet")
        python model_train.py --model ResNet --dataset cifar10 \
            --epochs=45 \
            --gpu=1 \
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
