#!/bin/bash

compression_strategy=$1
mode=$2


# Case logic for modes
case $mode in
    "search_lenet")
        python ../model_train.py --model LeNet --dataset mnist \
          --epochs=50 \
          --n_calls=10 \
          --k_fold=5 \
          --fullset=100 \
          --stop_patience=5 \
          --bayesian_search \
          --log=1 \
          --strategy="$compression_strategy"
        ;;

#      python ../model_train.py --model LeNet --dataset mnist \
#          --epochs=200 \
#          --n_calls=10 \
#          --k_fold=5 \
#          --fullset=1 \
#          --stop_patience=15 \
#          --bayesian_search \
#          --log=1 \
#          --strategy="$compression_strategy"
#        ;;

    "search_resnet18")
        python ../model_train.py --model ResNet18 --dataset cifar10 \
          --epochs=50 \
          --n_calls=10 \
          --k_fold=3 \
          --gpu=1 \
          --fullset=100 \
          --stop_patience=7 \
          --bayesian_search \
          --log=1 \
          --strategy="$compression_strategy"
        ;;

    "search_vgg11")
        python ../model_train.py --model VGG11 --dataset cifar10 \
          --epochs=50 \
          --n_calls=10 \
          --k_fold=3 \
          --gpu=1 \
          --fullset=100 \
          --stop_patience=7 \
          --bayesian_search \
          --log=1 \
          --strategy="$compression_strategy"
        ;;

    "l2_lenet")
        python ../model_train.py --model LeNet --dataset mnist \
            --epochs=45 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --lr_decay=3 \
            --log=1 \
            --train_on_baseline=2 \
            --strategy="$compression_strategy"
        ;;

    "baseline_l2_lenet")
        python ../model_train.py --model LeNet --dataset mnist \
            --epochs=45 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --train_on_baseline=1 \
            --lr_decay=3 \
            --gpu=0 \
            --log=1 \
            --strategy="$compression_strategy"
        ;;

    "baseline_l2_vgg11")
        python ../model_train.py --model VGG11 --dataset cifar10 \
            --epochs=40 \
            --gpu=1 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=20 \
            --train_on_baseline=1 \
            --lr_decay=3 \
            --log=1 \
            --strategy="$compression_strategy"
        ;;

    "no_l2_vgg11")
        python ../model_train.py --model VGG11 --dataset cifar10 \
            --epochs=50 \
            --gpu=1 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=15 \
            --train_on_baseline=0 \
            --lr_decay=3 \
            --log=1 \
            --strategy="$compression_strategy"
        ;;

    "no_l2_lenet")
        python ../model_train.py --model LeNet --dataset mnist \
            --epochs=45 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --train_on_baseline=0 \
            --lr_decay=3 \
            --log=1 \
            --strategy="$compression_strategy"
        ;;
    "no_l2_resnet18")
        python ../model_train.py --model ResNet18 --dataset cifar10 \
            --epochs=45 \
            --gpu=1 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --train_on_baseline=0 \
            --lr_decay=3 \
            --log=1 \
            --strategy="$compression_strategy"
        ;;

    "baseline_l2_resnet18")
        python ../model_train.py --model ResNet18 --dataset cifar10 \
            --epochs=40 \
            --gpu=1 \
            --k_fold=1 \
            --fullset=100 \
            --stop_patience=10 \
            --train_on_baseline=1 \
            --lr_decay=3 \
            --log=1 \
            --strategy="$compression_strategy"
        ;;
    *)
        echo "Invalid mode provided. Please use: search, training, baseline_l2, etc."
        exit 1
        ;;
esac
