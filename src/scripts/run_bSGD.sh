#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_l2_vgg11"

# LeNet: search_lenet  l2_lenet  baseline_l2_lenet  no_l2_lenet
# ResNet18: search_resnet18  no_l2_resnet18  baseline_l2_resnet18
# VGG11: baseline_l2_vgg11

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.01, "buckets": 5000, "sparse_buckets": 4999}'
base_strategy_resnet='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.01, "buckets": 10000, "sparse_buckets": 1}'
base_strategy_vgg11='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.05, "buckets": 10000, "sparse_buckets": 1}'

# LeNet
#buckets=(       1000)
#sparse_buckets=( 975)

# VGG11
buckets=(       100 2000 5000 10000)
sparse_buckets=( 99 1999 4999 9950)

# ResNet18
#buckets=(       2000 5000 10000)
#sparse_buckets=( 1999 4999 9950)

parallel=0

runs=2
for ((r=1; r<=runs; r++))
do
case $mode in
    "search_lenet"|"l2_lenet"|"baseline_l2_lenet"|"no_l2_lenet")
        for i in "${!buckets[@]}"; do
            bucket="${buckets[$i]}"
            sparse_bucket="${sparse_buckets[$i]}"

            modified_strategy=$(python -c "import json; strategy=json.loads('$base_strategy'); strategy['buckets']=$bucket; strategy['sparse_buckets']=$sparse_bucket; print(json.dumps(strategy))")

            if [ "$parallel" -eq 1 ]; then
                ./run_main.sh "$modified_strategy" "$mode" &
            else
                ./run_main.sh "$modified_strategy" "$mode"
            fi
        done
        ;;

    "search_resnet18"|"no_l2_resnet18"|"baseline_l2_resnet18")
        for i in "${!buckets[@]}"; do
            bucket="${buckets[$i]}"
            sparse_bucket="${sparse_buckets[$i]}"

            modified_strategy=$(python -c "import json; strategy=json.loads('$base_strategy_resnet'); strategy['buckets']=$bucket; strategy['sparse_buckets']=$sparse_bucket; print(json.dumps(strategy))")

            if [ "$parallel" -eq 1 ]; then
                ./run_main.sh "$modified_strategy" "$mode" &
            else
                ./run_main.sh "$modified_strategy" "$mode"
            fi
        done
        ;;

    "baseline_l2_vgg11")
        for i in "${!buckets[@]}"; do
            bucket="${buckets[$i]}"
            sparse_bucket="${sparse_buckets[$i]}"

            modified_strategy=$(python -c "import json; strategy=json.loads('$base_strategy_vgg11'); strategy['buckets']=$bucket; strategy['sparse_buckets']=$sparse_bucket; print(json.dumps(strategy))")

            if [ "$parallel" -eq 1 ]; then
                ./run_main.sh "$modified_strategy" "$mode" &
            else
                ./run_main.sh "$modified_strategy" "$mode"
            fi
        done
        ;;
    *)
        echo "Invalid mode provided. Please use: search, training, baseline_l2, or no_l2"
        exit 1
        ;;
esac
done