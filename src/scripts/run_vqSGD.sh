#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_l2_lenet"

# LeNet:    search_lenet      baseline_l2_lenet       l2_lenet        no_l2_lenet
# ResNet18: search_resnet18   baseline_l2_resnet18    no_l2_resnet18
# VGG11:    search_vgg11      baseline_l2_vgg11       no_l2_vgg11

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": K_VALUE}'
base_strategy_resnet='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": K_VALUE}'
base_strategy_vgg11='{"optimizer": "sgd", "compression": "vqsgd", "learning_rate": 0.01, "repetition": K_VALUE}'

repetitions=(250 500 1000 1500)
repetitions_vgg11=(30000)
repetitions_resnet=(20000 50000)

runs=3
for ((i=1; i<=runs; i++))
do
case $mode in
    "search_lenet"|"l2_lenet"|"baseline_l2_lenet"|"no_l2_lenet")
        for k in "${repetitions[@]}"; do
            modified_strategy="${base_strategy//K_VALUE/$k}"
            ./run_main.sh "$modified_strategy" "$mode"
        done
        ;;

    "search_resnet18"|"no_l2_resnet18"|"baseline_l2_resnet18")
        for k in "${repetitions_resnet[@]}"; do
            modified_strategy="${base_strategy_resnet//K_VALUE/$k}"
            ./run_main.sh "$modified_strategy" "$mode"
        done
        ;;

    "baseline_l2_vgg11")
        for k in "${repetitions_vgg11[@]}"; do
            modified_strategy="${base_strategy_vgg11//K_VALUE/$k}"
            ./run_main.sh "$modified_strategy" "$mode"
        done
    ;;

    *)
        echo "Invalid mode provided."
        exit 1
        ;;
esac
done