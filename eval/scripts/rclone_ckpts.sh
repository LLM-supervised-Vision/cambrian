#!/bin/bash

# Base directory and keys
HF_DIR_KEY="hf_paligemma"
HF_PALIGEMMA_CKPT_DIR="/data/austin"
GDRIVE_NAME="gdrive"

# List of MLLM keys
MLLM_KEYS=(
    # "stage_0_pt" "stage_0_sft"
    # "stage_1_pt" "stage_1_sft"
    # "stage_2_pt" "stage_2_sft"
    # "stage_3_pt" "stage_3_sft"
    
    # "stage_0_curriculum" "stage_0_no-curriculum"
    # "stage_1_curriculum" "stage_1_no-curriculum"
    # "stage_2_curriculum" "stage_2_no-curriculum"

    # stage_0_sft_paligemma-init
    # 1_sft_paligemma-PT-PT_paligemma-PT-PT_basic_clean
    1_sft_siglip-PT_G2B-fresh-fresh_baisc_clean

    # 1_sft_random-PT_G2B-fresh-fresh_advanced_clean
    # 1_sft_random-PT_G2B-fresh-fresh_basic_clean
    # 1_sft_random-PT_G2B-fresh-PT_basic_ann

    # 0_sft_siglip-PT-PT_G2B-fresh-fresh_basic_clean
    # 0_sft_siglip-PT_G2B-fresh-fresh_baisc_clean_all-loss
    # 0_sft_random-PT-PT_G2B-fresh-fresh-basic_clean

    # 0_pt_ocr-init 0_sft_ocr-init
    # 1_pt_ocr-init 1_sft_ocr-init

)

# Loop through each key and perform rclone
for MLLM_KEY in "${MLLM_KEYS[@]}"; do
    echo "Processing $MLLM_KEY..."

    # Define source and destination
    HF_PALIGEMMA_CKPT="$HF_PALIGEMMA_CKPT_DIR/bv2cambrian_ckpts/$HF_DIR_KEY-$MLLM_KEY"
    SOURCE="$GDRIVE_NAME:$HF_DIR_KEY/$MLLM_KEY"
    DESTINATION="$HF_PALIGEMMA_CKPT"

    # Create destination directory if it doesn't exist
    mkdir -p "$DESTINATION"

    # Perform rclone copy
    echo "Copying $SOURCE to $DESTINATION..."
    rclone copy -v --transfers 32 --checkers 32 --buffer-size 64M --drive-chunk-size 128M --drive-upload-cutoff 128M "$SOURCE" "$DESTINATION"

    # Check if rclone succeeded
    if [ $? -eq 0 ]; then
        echo "Successfully copied $MLLM_KEY."
    else
        echo "Failed to copy $MLLM_KEY."
    fi

    echo ""
done

echo "All keys processed."