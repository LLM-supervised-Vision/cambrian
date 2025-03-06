#!/bin/bash

# Official models to be evaluated:
# google/paligemma-3b-pt-224
# google/paligemma-3b-pt-896
# google/paligemma-3b-mix-224
# google/paligemma2-3b-pt-224
# google/paligemma2-3b-pt-896
# google/paligemma2-10b-pt-224
# google/paligemma2-10b-pt-896
# google/paligemma2-28b-pt-896

# Mine:
# /data/austin/bv2cambrian_ckpts/hf_paligemma-stage_0_pt
# /data/austin/bv2cambrian_ckpts/hf_paligemma-stage_0_sft
# /data/austin/bv2cambrian_ckpts/hf_paligemma-stage_1_pt
# /data/austin/bv2cambrian_ckpts/hf_paligemma-stage_1_sft

device=$1
model_path=$2
question_extension="Give the short answer directly."


# # Run the evaluation script
# CUDA_VISIBLE_DEVICES=$device bash /data/austin/cambrian/eval/scripts/run_all_benchmarks.sh \
#     $device \
#     "$model_path" \
#     plain \
#     "$question_extension"



# Run the evaluation script over HPC GREENE
bash /scratch/zw2526/workspace/cambrian/eval/slurm/submit_all_benchmarks_parallel.bash \
    --ckpt "$model_path" \
    --conv_mode "plain"