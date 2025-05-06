#!/bin/bash

# Base directory and keys
HF_DIR_KEY="hf_paligemma"
HF_PALIGEMMA_CKPT_DIR="/data/austin"
GDRIVE_NAME="gdrive"

# List of MLLM keys
MLLM_KEYS=(
    # 0_sft_p2-10b_mqa-concise_7068k-shuffled
    # 1_sft_p2-10b_mqa-concise_7068k-shuffled
    # 2_sft_p2-10b_mqa-concise_7068k-shuffled
    
    # 0_sft_p2-10b_mqa-concise_737k-shuffled
    # 1_sft_p2-10b_mqa-concise_737k-shuffled
    # 2_sft_p2-10b_mqa-concise_737k-shuffled
    
    # 0_sft_p2-10b_mqa-concise_5565k-shuffled
    # 1_sft_p2-10b_mqa-concise_5565k-shuffled
    # 1_sft_p2-10b_mqa-concise_5565k-shuffled_froz9
    # 1_sft_p2-10b_mqa-concise_5565k-shuffled_froz-llm-adap
    # 2_sft_p2-10b_mqa-concise_5565k-shuffled
    3_sft_p2-10b_mqa-concise_5565k-shuffled

    # 0_sft_p2-3b_mqa-concise_5565k-shuffled
    # 1_sft_p2-3b_mqa-concise_5565k-shuffled

    # 0_sft_p2-3b_mqa-concise_737k-shuffled
    # 1_sft_p2-3b_mqa-concise_737k-shuffled
    # 2_sft_p2-3b_mqa-concise_737k-shuffled

    # 0_sft_p2-3b_mqa-concise_10M

    # 1_sft_p2-10b_mqa-concise_10M
    # 2_sft_p2-10b_mqa-concise_10M
    
    # 1_sft_p2-10b_tqfa-concise_1.6M-5565k

    # 1_sft_p2-10b_tqfa-concise_1.6M-10M
    # 2_sft_p2-10b_tqfa-concise_1.6M-10M
    # 3_sft_p2-10b_tqfa-concise_1.6M-10M

    # 0_sft_p2-10b_tqa-concise_1.6M-737k
    # 1_sft_p2-10b_tqa-concise_1.6M-737k
    # 2_sft_p2-10b_tqa-concise_1.6M-737k
    # 3_sft_p2-10b_tqa-concise_1.6M-737k

    # 0_sft_p2-3b_tqa-concise_1.6M-737k
    # 1_sft_p2-3b_tqa-concise_1.6M-737k
    # 2_sft_p2-3b_tqa-concise_1.6M-737k
    # 3_sft_p2-3b_tqa-concise_1.6M-737k

    # 1_sft_p2-3b-448-pp224
    # 2_sft_p2-3b-448-pp224
    # 3_sft_p2-3b-448-pp224

    # 0_sft_p2-10b-224
    # 1_sft_p2-10b-224

    # 1_sft_p2-3b-224_corrected-pp
    # 2_sft_p2-3b-224_corrected-pp

    # 1_sft_p2-3b-224_split-qa
    # 2_sft_p2-3b-224_split-qa

    # 2_sft_p2-3b-224_split-qa_mix1.0-1.0
    # 2_sft_p2-3b-224_split-qa_mix0.25-1.0

    # 0_sft_p2-3b-448-448_split-qa
    # 0_sft_p2-3b-448-224_split-qa
    # 1_pt_p2-3b-448-224_split-qa
    # 1_sft_p2-3b-448-224_split-qa
    # 2_sft_p2-3b-448-224_split-qa
    # 3_sft_p2-3b-448-224_split-qa

    # 0_sft_p2-3b-224
    # 1_sft_p2-3b-224
    # 2_sft_p2-3b-224

    # 2_sft_p3b224_wd0.0_re-init
    # 2_sft_p3b224_wd0.0_inherit
    # 3_sft_p3b224_wd0.0_re-init
    # 3_sft_p3b224_wd0.0_inherit

    # "stage_0_pt" "stage_0_sft"
    # "stage_1_pt" "stage_1_sft"
    # "stage_2_pt" "stage_2_sft"
    # "stage_3_pt" "stage_3_sft"
    
    # "stage_0_curriculum" "stage_0_no-curriculum"
    # "stage_1_curriculum" "stage_1_no-curriculum"
    # "stage_2_curriculum" "stage_2_no-curriculum"

    # stage_0_sft_paligemma-init
    # 1_sft_paligemma-PT-PT_paligemma-PT-PT_basic_clean
    # 1_sft_siglip-PT_G2B-fresh-fresh_baisc_clean

    # # 0_SFT_freeze_bs512_1e-5-F_2e-5-F_2e-5
    # # 0_SFT_freeze_bs512_1e-5-F_2e-5_2e-5
    # # 0_SFT_unfreeze_bs512_1e-5_2e-5_2e-5
    # 0_SFT_freeze_bs4k_1e-5-F_2e-5-F_2e-5
    # 0_SFT_freeze_bs4k_1e-5-F_2e-5_2e-5
    # 0_SFT_unfreeze_bs4k_1e-5_2e-5_2e-5

    # 0_sft_p3b
    # 1_sft_p3b
    # 2_sft_p3b
    # 3_sft_p3b

    # 1_SFT_bs16k_1e-5_2e-5_2e-5_10ep_single
    # 1_SFT_bs16k_1e-5_2e-5_2e-5_10ep_wd0_single
    # 1_SFT_bs16k_1e-5_2e-5_2e-5_1ep_single
    # 1_SFT_bs16k_1e-5_2e-5_2e-5_4ep_multi_txtlen256
    # 1_SFT_bs16k_1e-5_2e-5_2e-5_1ep_multi_txtlen128
    # 1_SFT_bs512_1e-5_2e-5_2e-5_1ep_multi_txtlen128
    # 1_SFT_bs4k_1e-5_2e-5_2e-5_1ep_multi_txtlen256
    # 1_SFT_bs512_1e-5_2e-5_2e-5_1ep_multi_txtlen256
    # 1_SFT_bs512_1e-5_2e-5_2e-5_wd0_1ep_multi_txtlen256
    # 1_SFT_bs512_5e-6_1e-5_1e-5_1ep_multi_txtlen256

    # 1_combined_bs4k_1ep-1.0-1.0
    # 1_combined_bs512_1ep-1.0-1.0

    # 1_sft_random-PT_G2B-fresh-fresh_advanced_clean
    # 1_sft_random-PT_G2B-fresh-fresh_basic_clean
    # 2_sft_random-PT_G2B-fresh-fresh_basic_clean
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