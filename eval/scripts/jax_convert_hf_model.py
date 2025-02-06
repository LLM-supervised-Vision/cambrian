import os
import json
import torch
import struct
import argparse
import numpy as np

import transformers
from huggingface_hub import login
import os
hf_token = os.getenv('HF_TOKEN')
login(token=hf_token)

def convert_void_array(arr):
    """Convert void array to float32."""
    # Convert void array to bytes
    flat_bytes = arr.tobytes()
    # Interpret each 2-byte chunk as a bfloat16
    num_elements = len(flat_bytes) // 2
    floats = []
    
    for i in range(num_elements):
        # Extract 2 bytes and convert to float32
        two_bytes = flat_bytes[i*2:(i+1)*2]
        # Pad to 4 bytes (float32)
        padded = two_bytes + b'\x00\x00'
        # Convert to float32
        val = struct.unpack('f', padded)[0]
        floats.append(val)
    
    # Reshape to original shape
    return np.array(floats, dtype=np.float32).reshape(arr.shape)

def main(args):

    print(f"Loading JAX checkpoint from {args.jax_ckpt}")
    npz_path = os.path.join(args.jax_ckpt,"hf_state_dict.npz")
    np_file = np.load(npz_path)

    print(f"Converting JAX checkpoint dtype to float32")
    params = {}
    for k, v in np_file.items():
        if v.dtype == np.dtype('|V2'):
            print(f"Converting {k} from |V2 to float32")
            params[k] = torch.tensor(convert_void_array(v))
        else:
            params[k] = torch.tensor(v)

    # check params
    print(f"Verifying all params are float32")
    for k, v in params.items():
        if v.dtype != torch.float32:
            print(k, v.dtype)

    # Load config and model
    config_path = os.path.join(os.path.dirname(npz_path), "hf_config.json")
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = transformers.PaliGemmaConfig(**config_dict)
    model = transformers.PaliGemmaForConditionalGeneration(config)
    print(f"Loaded model from config: {config}")


    for k,v in model.state_dict().items():
        if not k in params:
            print(f"WARNING: {k} not found in JAX checkpoint")
            continue
     

    print("Loading state dict into model")
    missing, unexpected = model.load_state_dict(params, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jax_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_id', type=str, default="google/paligemma-3b-pt-224")
    args = parser.parse_args()
    main(args)

