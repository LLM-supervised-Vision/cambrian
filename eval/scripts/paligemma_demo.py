from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/paligemma2-3b-pt-224"

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
import pdb; pdb.set_trace()
processor = AutoProcessor.from_pretrained(model_id)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)
image_path = "/data/austin/big_vision/example_images/ocr_test.jpg"
image = Image.open(image_path).convert("RGB")

# Instruct the model to create a caption in Spanish
prompt = "What is the dividend payout in 2012?"
model_inputs = processor(text=prompt, images=image, return_tensors="pt")
import pdb; pdb.set_trace()
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
