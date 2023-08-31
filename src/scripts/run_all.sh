#!/bin/bash

# Kill
# pkill -f run_EFsignSGD.sh
# pkill -f run_all.sh

# Master script
DEFAULT_MODE="baseline_l2_vgg11"

# LeNet: search_lenet  l2_lenet  baseline_l2_lenet  no_l2_lenet
# ResNet18: search_resnet18  no_l2_resnet18  baseline_l2_resnet18
# VGG11: search_vgg11  baseline_l2_vgg11

mode=${1:-$DEFAULT_MODE}

scripts=(
    "run_EFsignSGD.sh"
    #"run_FetchSGD.sh"
    #"run_GradientSparsification.sh"
    #"run_memSGD.sh"
    #"run_NaturalCompression.sh"
    "run_OneBitSGD.sh"
    #"run_SGD.sh"
    #"run_SparseGradient.sh"
    #"run_TernGrad.sh"
    #"run_TopK.sh"
    #"run_vqSGD.sh"
)

pids=()

# This function will be called when you interrupt the script
stop_scripts() {
    echo "Shutting down all subprocesses..."
    for pid in "${pids[@]}"; do
        kill $pid
    done
}

# Set the trap
trap stop_scripts SIGINT SIGTERM

execute_scripts() {
    for script in "${scripts[@]}"; do
        ./$script $1 &
        pids+=($!)
        echo "$pids"
    done
    wait
}

execute_scripts "$mode"

