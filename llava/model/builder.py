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


import os, sys
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

sys.path.append('/mnt/haiyangguo/mywork/CL-MLLM/LLaVA-HiDe')

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", text_tower=None, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

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

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            base_cfg_pretrained = AutoConfig.from_pretrained(model_base)
            for attr_name in (
                "mm_projector_type",
                "mm_hidden_size",
                "mm_vision_select_layer",
                "mm_use_im_start_end",
                "mm_use_im_patch_token",
                "image_aspect_ratio",
            ):
                if hasattr(base_cfg_pretrained, attr_name):
                    setattr(
                        lora_cfg_pretrained,
                        attr_name,
                        getattr(base_cfg_pretrained, attr_name),
                    )
            if text_tower is not None:
                lora_cfg_pretrained.mm_text_tower = text_tower
                lora_cfg_pretrained.mm_vision_tower = text_tower
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)

            clip_tokenizer = AutoTokenizer.from_pretrained(
                text_tower,
                cache_dir=None,
                model_max_length=77,
                padding_side="right",
                use_fast=True,
            )

            model.set_clip_tokenizer(clip_tokenizer)
            model.set_tokenizer(tokenizer)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
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
            if load_4bit or load_8bit:
                non_lora_trainables = {
                    k: v for k, v in non_lora_trainables.items()
                    if not k.endswith('lm_head.weight')
                }
            model_state = model.state_dict()
            mismatched_keys = [
                k for k, v in non_lora_trainables.items()
                if k in model_state and model_state[k].shape != v.shape
            ]
            if mismatched_keys:
                warnings.warn(
                    "Skipping non-LoRA weights with incompatible shapes: "
                    + ", ".join(
                        f"{k}: checkpoint={tuple(non_lora_trainables[k].shape)} "
                        f"model={tuple(model_state[k].shape)}"
                        for k in mismatched_keys
                    )
                )
                non_lora_trainables = {
                    k: v for k, v in non_lora_trainables.items()
                    if k not in mismatched_keys
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from HiDe.peft import PeftModel, TaskType, get_peft_model, HiDeMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict
            # else:
            #     from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            consensus_path = os.path.join(model_path, "consensus_subspaces.pt")
            if os.path.exists(consensus_path):
                model.load_consensus_state(
                    torch.load(consensus_path, map_location="cpu")
                )
                print(f"Loaded consensus subspaces from {consensus_path}")
            adapter_config = model.peft_config["default"]
            if hasattr(adapter_config, "cur_task"):
                model.set_cur_task(
                    adapter_config.cur_task, adapter_config.expert_num
                )
            print('Keeping HiDe LoRA experts active for inference...')
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
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
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        input_embeddings = model.get_input_embeddings()
        if input_embeddings.weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        if load_4bit and device == "cuda":
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None and not hasattr(output_embeddings.weight, "quant_state"):
                fp16_lm_head = torch.nn.Linear(
                    output_embeddings.in_features,
                    output_embeddings.out_features,
                    bias=False,
                    device=device,
                    dtype=torch.float16,
                )
                index_path = os.path.join(model_base, "pytorch_model.bin.index.json")
                if os.path.exists(index_path):
                    import json
                    with open(index_path, "r") as handle:
                        weight_map = json.load(handle)["weight_map"]
                    shard_name = weight_map.get("lm_head.weight")
                    if shard_name is not None:
                        shard = torch.load(os.path.join(model_base, shard_name), map_location="cpu")
                        fp16_lm_head.weight.data.copy_(shard["lm_head.weight"].to(device=device, dtype=torch.float16))
                        del shard
                    else:
                        warnings.warn("lm_head.weight was not found in the base model index; 4-bit eval may produce invalid logits.")
                else:
                    warnings.warn("Base model shard index was not found; 4-bit eval may produce invalid logits.")
                model.set_output_embeddings(fp16_lm_head)

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

        text_tower = model.get_text_tower()
        if not text_tower.is_loaded:
            text_tower.load_model()
        text_tower.to(device=device, dtype=torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
