#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_l2_vgg11"

# LeNet: search_lenet  l2_lenet  baseline_l2_lenet  no_l2_lenet
# ResNet18: search_resnet18  no_l2_resnet18  baseline_l2_resnet18
# VGG11: baseline_l2_vgg11

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.01, "c": C_VALUE, "r": 1,
                "topk": C_VALUE, "momentum": 0.9}'

base_strategy_resnet='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.01, "c": C_VALUE, "r": 1,
                      "topk": C_VALUE, "momentum": 0.9}'

base_strategy_vgg11='{"optimizer": "fetchsgd", "compression": "none", "learning_rate": 0.01, "c": C_VALUE, "r": 1,
                      "topk": C_VALUE, "momentum": 0.9}'

counters=(1000 2000 5000)
counters_resnet=(5000 10000 100000)
counters_vgg11=(100000 500000)

runs=3
for ((i=1; i<=runs; i++))
do
case $mode in
    "search_lenet"|"l2_lenet"|"baseline_l2_lenet"|"no_l2_lenet")
        for c in "${counters[@]}"; do
            modified_strategy="${base_strategy//C_VALUE/$c}"
            ./run_main.sh "$modified_strategy" "$mode"
        done
        ;;

    "search_resnet18"|"no_l2_resnet18"|"baseline_l2_resnet18")
        for c in "${counters_resnet[@]}"; do
            modified_strategy="${base_strategy_resnet//C_VALUE/$c}"
            ./run_main.sh "$modified_strategy" "$mode"
        done
        ;;

    "baseline_l2_vgg11")
        for c in "${counters_vgg11[@]}"; do
            modified_strategy="${base_strategy_vgg11//C_VALUE/$c}"

            ./run_main.sh "$modified_strategy" "$mode"
            done
            ;;

    *)
        echo "Invalid mode provided."
        exit 1
        ;;
esac
done





