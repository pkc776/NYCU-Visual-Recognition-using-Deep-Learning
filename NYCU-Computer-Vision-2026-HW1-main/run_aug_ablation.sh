#!/bin/bash

# A simple ablation script to iterate over different augmentation policies
EPOCHS=30
BATCH_SIZE=128
AUG_LIST=("randaug" "autoaug" "colorjitter")

echo "Starting Augmentation Ablation Study..."
echo "Comparing: ${AUG_LIST[*]} on 3 GPUs simultaneously"
echo ""

# Ensure the output directory exists
mkdir -p ./checkpoints_ablation

for i in "${!AUG_LIST[@]}"; do
    AUG="${AUG_LIST[$i]}"
    GPU_ID=$((i % 3))
    
    echo "=========================================="
    echo "Starting training: $AUG on GPU $GPU_ID "
    echo "Log file: ./checkpoints_ablation/train_${AUG}.log"
    echo "=========================================="
    
    # Configure the save directory to be distinct for each run
    SAVE_DIR="./checkpoints_ablation/$AUG"
    
    # Run training in the background on the specific GPU, piping output to a log file
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/train.py \
        --data_dir ./data \
        --save_dir "$SAVE_DIR" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --aug_type "$AUG" > "./checkpoints_ablation/train_${AUG}.log" 2>&1 &
done

echo "All tasks are running in the background. Waiting for them to finish..."

# Wait for all background jobs to finish
wait

echo "Ablation study completed."
echo "Checkpoints and logs are stored in ./checkpoints_ablation/."
