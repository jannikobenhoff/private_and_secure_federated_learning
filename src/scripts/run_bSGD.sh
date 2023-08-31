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
base_strategy_vgg11='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.01, "buckets": 10000, "sparse_buckets": 1}'

buckets=(       10000 2000 5000)
sparse_buckets=( 9950 1999 4999)

#buckets=(      1000 100 50) #1000 100 1000 100 1000 100)
#sparse_buckets=( 900 90 49) #999 99 999 99 999 99)

# best for resnet was 1000/9000

parallel=0

runs=3
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