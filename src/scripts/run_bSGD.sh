#!/bin/bash

# Default mode set at the top of the script
DEFAULT_MODE="baseline_l2"  # search  training  baseline_l2  no_l2  no_l2_resnet  baseline_resnet

# If an argument is provided, use it. Otherwise, use the default.
mode=${1:-$DEFAULT_MODE}

base_strategy='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.01, "buckets": 5000, "sparse_buckets": 4999}'

base_strategy_resnet='{"optimizer": "sgd", "compression": "bsgd", "learning_rate": 0.1, "buckets": 10000, "sparse_buckets": 1}'

buckets=(       10000 2000 5000)
sparse_buckets=( 9950 1999 4999)

case $mode in
    "search")
        for i in "${!buckets[@]}"; do
            bucket="${buckets[$i]}"
            sparse_bucket="${sparse_buckets[$i]}"

            modified_strategy=$(python -c "import json; strategy=json.loads('$base_strategy'); strategy['buckets']=$bucket; strategy['sparse_buckets']=$sparse_bucket; print(json.dumps(strategy))")

            python ../model_train.py --model LeNet --dataset mnist \
              --epochs=100 \
              --n_calls=10 \
              --k_fold=5 \
              --fullset=1 \
              --stop_patience=10 \
              --bayesian_search \
              --log=1 \
              --strategy="$modified_strategy"
        done
        ;;

    "training")
        for i in "${!buckets[@]}"; do
            bucket="${buckets[$i]}"
            sparse_bucket="${sparse_buckets[$i]}"

            modified_strategy=$(python -c "import json; strategy=json.loads('$base_strategy'); strategy['buckets']=$bucket; strategy['sparse_buckets']=$sparse_bucket; print(json.dumps(strategy))")
          python ../model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=2 \
                --lr_decay=3 \
                --log=1 \
                --gpu=0 \
              --strategy="$modified_strategy"
        done
        ;;

    "baseline_l2")
        for i in "${!buckets[@]}"; do
            bucket="${buckets[$i]}"
            sparse_bucket="${sparse_buckets[$i]}"

            modified_strategy=$(python -c "import json; strategy=json.loads('$base_strategy'); strategy['buckets']=$bucket; strategy['sparse_buckets']=$sparse_bucket; print(json.dumps(strategy))")
          python ../model_train.py --model LeNet --dataset mnist \
                --epochs=45 \
                --k_fold=1 \
                --fullset=100 \
                --stop_patience=10 \
                --train_on_baseline=1 \
                --lr_decay=3 \
                --log=1 \
                --gpu=0 \
              --strategy="$modified_strategy"
        done
        ;;

    "no_l2_resnet")
        for i in "${!buckets[@]}"; do
                bucket="${buckets[$i]}"
                sparse_bucket="${sparse_buckets[$i]}"

                modified_strategy=$(python -c "import json; strategy=json.loads('$base_strategy_resnet'); strategy['buckets']=$bucket; strategy['sparse_buckets']=$sparse_bucket; print(json.dumps(strategy))")

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
        done
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
