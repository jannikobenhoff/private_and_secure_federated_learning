#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_l2_lenet"

# LeNet:    search_lenet      baseline_l2_lenet       l2_lenet        no_l2_lenet
# ResNet18: search_resnet18   baseline_l2_resnet18    no_l2_resnet18
# VGG11:    search_vgg11      baseline_l2_vgg11       no_l2_vgg11

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

runs=1
for ((i=1; i<=runs; i++))
do
    case $mode in
        "search_lenet"|"l2_lenet"|"baseline_l2_lenet"|"no_l2_lenet")
            ./run_main.sh '{"compression": "naturalcompression"}' "$mode"
        ;;

        "search_resnet18"|"no_l2_resnet18"|"baseline_l2_resnet18")
            ./run_main.sh '{"compression": "naturalcompression"}' "$mode"
        ;;

        "baseline_l2_vgg11")
            ./run_main.sh '{"compression": "naturalcompression"}' "$mode"
        ;;
        *)
            echo "Invalid mode provided."
            exit 1
            ;;
    esac
done