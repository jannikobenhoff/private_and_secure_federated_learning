#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_l2_resnet18"

# LeNet:    search_lenet      baseline_l2_lenet       l2_lenet        no_l2_lenet
# ResNet18: search_resnet18   baseline_l2_resnet18    no_l2_resnet18
# VGG11:    search_vgg11      baseline_l2_vgg11       no_l2_vgg11

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "atomo", "learning_rate": 0.01, "svd_rank":RANK}'
base_strategy_resnet='{"optimizer": "sgd", "compression": "atomo", "learning_rate": 0.001, "svd_rank":RANK}'
base_strategy_vgg11='{"optimizer": "sgd", "compression": "atomo", "learning_rate": 0.05, "svd_rank":RANK}'

ranks=(1 3 6)
#ranks=(6 6)

parallel=0
runs=1
for ((i=1; i<=runs; i++))
do
    case $mode in
        "search_lenet"|"l2_lenet"|"baseline_l2_lenet"|"no_l2_lenet")
            for rank in "${ranks[@]}"; do
            modified_strategy="${base_strategy//RANK/$rank}"

            if [ "$parallel" -eq 1 ]; then
                    ./run_main.sh "$modified_strategy" "$mode" &
                else
                    ./run_main.sh "$modified_strategy" "$mode"
            fi
            done
            ;;

        "search_resnet18"|"no_l2_resnet18"|"baseline_l2_resnet18")
            for rank in "${ranks[@]}"; do
            modified_strategy="${base_strategy_resnet//RANK/$rank}"

            if [ "$parallel" -eq 1 ]; then
                    ./run_main.sh "$modified_strategy" "$mode" &
                else
                    ./run_main.sh "$modified_strategy" "$mode"
            fi
            done
            ;;
        "baseline_l2_vgg11")
            for rank in "${ranks[@]}"; do
            modified_strategy="${base_strategy_vgg11//RANK/$rank}"

            if [ "$parallel" -eq 1 ]; then
                    ./run_main.sh "$modified_strategy" "$mode" &
                else
                    ./run_main.sh "$modified_strategy" "$mode"
            fi
            done
            ;;
        *)
            echo "Invalid mode provided."
            exit 1
            ;;
    esac
done