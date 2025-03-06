#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from cambrian.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from ezcolorlog import root_logger as logger

from cambrian.model.language_model.cambrian_llama import CambrianLlamaForCausalLM
from cambrian.model.language_model.cambrian_mistral import CambrianMistralForCausalLM


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    image_processor = None

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'cambrian' in model_name.lower():
        # Load Cambrian model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from cambrian.model.language_model.cambrian_llama import CambrianConfig
            lora_cfg_pretrained = CambrianConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            logger.info('Loading Cambrian from base model...')
            model = CambrianLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            logger.info('Loading additional Cambrian weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            logger.info('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            logger.info('Merging LoRA weights...')
            model = model.merge_and_unload()
            logger.info('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            logger.info(f'Loading Cambrian-1 from base model... {model_base}')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = CambrianLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = CambrianMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=False,
                    **kwargs
                )
            elif 'phi3' in model_name.lower():
                from cambrian.model.language_model.cambrian_phi3 import CambrianPhi3ForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = CambrianPhi3ForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=False,
                    **kwargs
                )
            elif 'gemma' in model_name.lower():
                from cambrian.model.language_model.cambrian_gemma import CambrianGemmaForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = CambrianGemmaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=False,
                    **kwargs
                )
            else:
                logger.info(f'Loading Cambrian from {model_path}')
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = CambrianLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            elif 'paligemma' in model_name.lower():
                from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, PaliGemmaConfig
                kwargs['device_map'] = 'cuda'
                # model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, **kwargs)
                model = EvalCompatiblePaliGemma.from_pretrained(model_path, **kwargs)
                # model.config.vocab_size = model.config.text_config.vocab_size
                # model.config.mm_vision_tower_aux_list = [os.path.join(model_path, 'bv_siglip_gemma_stage_0_pt.npz'),]
                # model.config.mm_vision_tower_aux_token_len_list = [196,]

                model.config.mm_use_im_start_end = False
                model.config.mm_use_im_patch_token = False

                # model.config.mm_vision_select_feature = "patch"
                # model.config.mm_vision_select_layer = -1

                # model.config.query_num_list = None
                # model.config.mm_projector_type = 'linear'
                # model.config.mm_hidden_size = model.config.vision_config.hidden_size
                model.config.image_token_len = model.config.vision_config.num_image_tokens
                model.config.mm_image_size = model.vision_tower.vision_model.embeddings.patch_embedding.kernel_size[0]*int(model.config.image_token_len**0.5)

                processor = PaliGemmaProcessor.from_pretrained(model_path)
                tokenizer = processor.tokenizer
                image_processor = [processor.image_processor]
                image_processor[0].crop_size = {"height": model.config.mm_image_size, "width": model.config.mm_image_size}

            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)


    if 'cambrian' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(model.config.vocab_size)# len(tokenizer))

        vision_tower_aux_list = model.get_vision_tower_aux_list()

        for vision_tower_aux in vision_tower_aux_list:
            if not vision_tower_aux.is_loaded:
                vision_tower_aux.load_model(device_map=device_map)
            vision_tower_aux.to(device=device, dtype=torch.float16)

        image_processor = [vision_tower_aux.image_processor for vision_tower_aux in vision_tower_aux_list]

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len



from transformers import PaliGemmaForConditionalGeneration
import torch

class EvalCompatiblePaliGemma(PaliGemmaForConditionalGeneration):
    """
    A subclass of PaliGemmaForConditionalGeneration that overrides the generate method
    to make it compatible with the inputs provided by the evaluation script.
    """
    
    def generate(
        self,
        input_ids=None,
        images=None,
        image_sizes=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        inputs_embeds=None,
        do_sample=None,
        temperature=None,
        top_p=None,
        num_beams=None,
        max_new_tokens=None,
        use_cache=None,
        **kwargs
    ):
        """
        Overridden generate method that accepts 'images' parameter 
        instead of 'pixel_values' expected by the parent class.
        """
        # Convert images to pixel_values if needed
        pixel_values = None
        if images is not None:
            # The model expects pixel_values instead of images
            pixel_values = images[0]

        # Replace token 109 (\n\n) with 108 at the end if present
        assert input_ids.size(0) == 1, "Batch size must be 1"
        if input_ids[0][-1] == 109:
            input_ids[0][-1] = 108

        sequence_to_replace = torch.tensor([-200, 108], device=input_ids.device)
        sequence_length = len(sequence_to_replace)

        # Find the starting index of the sequence
        start_index = None
        for i in range(len(input_ids[0]) - sequence_length + 1):
            if torch.equal(input_ids[0][i:i + sequence_length], sequence_to_replace):
                start_index = i
                break

        if start_index is None:
            raise ValueError("Sequence [-200, 108] not found in input_ids.")

        # Step 2: Create the replacement tensor
        replacement = torch.cat([
            torch.full((self.config.image_token_len,), self.config.image_token_index, dtype=input_ids.dtype, device=input_ids.device),
            torch.tensor([self.config.bos_token_id], dtype=input_ids.dtype, device=input_ids.device)
        ])

        # Step 3: Reconstruct the tensor
        new_input_ids = torch.cat([
            input_ids[:, :start_index],       # Part before [-200, 108]
            replacement.unsqueeze(0),         # Replacement sequence
            input_ids[:, start_index + sequence_length:]  # Part after [-200, 108]
        ], dim=1)

        # from transformers import PaliGemmaProcessor
        # processor = PaliGemmaProcessor.from_pretrained(self.config._name_or_path)
        # system_prompt = new_input_ids[:, :start_index]
        # specific_prompt = new_input_ids[:,start_index+self.config.image_token_len:]
        # print(f"system_prompt: {system_prompt} \nwhich means: {processor.batch_decode(system_prompt, skip_special_tokens=True)}")
        # print(f"specific_prompt: {specific_prompt} \nwhich means: {processor.batch_decode(specific_prompt, skip_special_tokens=True)}")

        # Call the parent's generate method with the converted inputs
        full_outputs = super().generate(
            input_ids=new_input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            **kwargs
        )

        new_input_length = new_input_ids.shape[1]
        return full_outputs[:, new_input_length:]