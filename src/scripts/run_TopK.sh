#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_l2_lenet"

# LeNet:    search_lenet      baseline_l2_lenet       l2_lenet        no_l2_lenet
# ResNet18: search_resnet18   baseline_l2_resnet18    no_l2_resnet18
# VGG11:    search_vgg11      baseline_l2_vgg11       no_l2_vgg11

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"compression": "topk", "k": K_VALUE}'
base_strategy_resnet='{"compression": "topk", "k": K_VALUE}'
base_strategy_vgg11='{"compression": "topk", "k": K_VALUE}'

top_ks=(6000 100 500 1000)
top_ks_resnet=(2500000) #100000
top_ks_vgg11=(1000 5000 10000) # 2000

parallel=0
runs=1
for ((i=1; i<=runs; i++))
do
case $mode in
    "search_lenet"|"l2_lenet"|"baseline_l2_lenet"|"no_l2_lenet")
        for k in "${top_ks[@]}"; do
            modified_strategy="${base_strategy//K_VALUE/$k}"
            if [ "$parallel" -eq 1 ]; then
                    ./run_main.sh "$modified_strategy" "$mode" &
                else
                    ./run_main.sh "$modified_strategy" "$mode"
            fi
        done
        ;;

    "search_resnet18"|"no_l2_resnet18"|"baseline_l2_resnet18")
        for k in "${top_ks_resnet[@]}"; do
            modified_strategy="${base_strategy_resnet//K_VALUE/$k}"
            if [ "$parallel" -eq 1 ]; then
                    ./run_main.sh "$modified_strategy" "$mode" &
                else
                    ./run_main.sh "$modified_strategy" "$mode"
            fi
        done
        ;;

    "baseline_l2_vgg11")
        for k in "${top_ks_vgg11[@]}"; do
            modified_strategy="${base_strategy_vgg11//K_VALUE/$k}"
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