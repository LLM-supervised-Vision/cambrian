#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <gcs_path>"
    exit 1
fi

# Get the GCS path from command line argument
GCS_PATH=$1

# Set evaluation directory
EVAL_DIR=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))
JAX_CKPT_DIR="$EVAL_DIR/jax_ckpts"

# Create directories if they don't exist
mkdir -p "$JAX_CKPT_DIR"

# Extract the last part of the GCS path as the checkpoint name
CKPT_NAME=$(basename "$GCS_PATH")

# Copy files from GCS to local directory if they don't exist locally
if [ ! -f "$JAX_CKPT_DIR/hf_config.json" ]; then
    gsutil -m cp "$GCS_PATH/hf_config.json" "$JAX_CKPT_DIR/"
fi

# Copy NPZ files if they don't exist locally
if [ ! -f "$JAX_CKPT_DIR"/*.npz ]; then
    gsutil -m cp "$GCS_PATH/*.npz" "$JAX_CKPT_DIR/"
fi

# Run the conversion script
python "$EVAL_DIR/jax_mllm_eval_hpc/jax_convert_hf_model.py" \
    --jax_ckpt "$JAX_CKPT_DIR" \
    --output_dir "$JAX_CKPT_DIR" \
    --model_id google/paligemma-3b-pt-224
