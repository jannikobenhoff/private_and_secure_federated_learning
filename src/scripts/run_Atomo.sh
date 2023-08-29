#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="l2_lenet"

# LeNet: search_lenet  l2_lenet  baseline_l2_lenet  no_l2_lenet
# ResNet18: search_resnet18  no_l2_resnet18  baseline_l2_resnet18
# VGG11: baseline_l2_vgg11

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "atomo", "learning_rate": 0.01, "svd_rank":RANK}'
base_strategy_resnet='{"optimizer": "sgd", "compression": "atomo", "learning_rate": 0.1, "svd_rank":RANK}'
base_strategy_vgg11='{"optimizer": "sgd", "compression": "atomo", "learning_rate": 0.1, "svd_rank":RANK}'

# ranks=(1 3 4)
ranks=(6 6)

case $mode in
    "search_lenet"|"l2_lenet"|"baseline_l2_lenet"|"no_l2_lenet")
        for rank in "${ranks[@]}"; do
        modified_strategy="${base_strategy//RANK/$rank}"

        ./run_main.sh "$modified_strategy" "$mode"
        done
        ;;

    "search_resnet18"|"no_l2_resnet18"|"baseline_l2_resnet18")
        for rank in "${ranks[@]}"; do
        modified_strategy="${base_strategy_resnet//RANK/$rank}"

        ./run_main.sh "$modified_strategy" "$mode"
        done
        ;;
    "baseline_l2_vgg11")
        for rank in "${ranks[@]}"; do
        modified_strategy="${base_strategy_vgg11//RANK/$rank}"

        ./run_main.sh "$modified_strategy" "$mode"
        done
        ;;
    *)
        echo "Invalid mode provided."
        exit 1
        ;;
esac
