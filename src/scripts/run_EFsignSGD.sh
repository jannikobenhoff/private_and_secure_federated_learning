#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_resnet"  # search  training  baseline_l2  no_l2  no_l2_resnet  baseline_resnet

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "efsignsgd", "compression": "none", "learning_rate": 0.01}'

base_strategy_resnet='{"optimizer": "efsignsgd", "compression": "none", "learning_rate": 0.1}'

case $mode in
    "search")
        python ../model_train.py --model LeNet --dataset mnist \
          --epochs=200 \
          --n_calls=10 \
          --k_fold=5 \
          --fullset=10 \
          --stop_patience=50 \
          --bayesian_search \
          --log=2 \
          --strategy="$base_strategy"
        ;;

    "training")
        python ../model_train.py --model LeNet --dataset mnist \
            --epochs=45 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --lr_decay=3 \
            --log=2 \
            --train_on_baseline=2 \
            --strategy="${base_strategy//CLIP_VALUE/$clip}"
        ;;

    "baseline_l2")
        python ../model_train.py --model LeNet --dataset mnist \
            --epochs=45 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --train_on_baseline=1 \
            --lr_decay=3 \
            --log=2 \
            --strategy="$base_strategy"
        ;;

    "no_l2")
        python ../model_train.py --model LeNet --dataset mnist \
            --epochs=45 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --train_on_baseline=0 \
            --lr_decay=3 \
            --log=2 \
            --strategy="$base_strategy"
        ;;

    "no_l2_resnet")
        python ../model_train.py --model ResNet --dataset cifar10 \
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

    "baseline_resnet")
        python ../model_train.py --model ResNet --dataset cifar10 \
            --epochs=45 \
            --gpu=1 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --train_on_baseline=1 \
            --lr_decay=3 \
            --log=1 \
            --strategy="$base_strategy_resnet"
        ;;

    *)
        echo "Invalid mode provided. Please use: search, training, baseline_l2, or no_l2"
        exit 1
        ;;
esac
