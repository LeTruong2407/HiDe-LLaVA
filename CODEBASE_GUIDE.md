# HiDe-LLaVA Codebase Guide

This document explains the structure and runtime logic of the `HiDe-LLaVA` repository so it is easier to read, modify, and debug.

## 1. What this repository is

`HiDe-LLaVA` is a fork of the LLaVA training/inference stack with a **HiDe-specific continual instruction tuning method** added on top.

At a high level:

- Base model: LLaVA 1.5 style multimodal causal LM
- Vision backbone: CLIP vision encoder
- Text guide backbone: CLIP text encoder
- Adaptation method: a custom PEFT fork with **HiDe MoE-LoRA**
- Training regime: sequential task-by-task continual instruction tuning
- Evaluation: task-specific generation scripts plus task-specific scoring scripts

The code is mostly composed of:

1. Upstream / lightly modified LLaVA code
2. A local PEFT fork under `HiDe/peft`
3. HiDe-specific routing / anchor logic inside the multimodal model path
4. Shell scripts that define the actual continual training schedule

---

## 2. Top-level layout

### Core directories

- `llava/`: main model, training, serving, and evaluation code
- `HiDe/`: local fork of `peft` with HiDe MoE-LoRA support
- `scripts/HiDe/`: task-by-task training and evaluation entrypoints
- `figure/`: paper figures
- `config.json`: replacement model config for LLaVA base weights
- `eval.py`: local COCO evaluation helper replacement mentioned in the README
- `test_CKA_sim.py`: analysis script for layer similarity / CKA

### Mental model

If you want to understand the repository quickly, read it in this order:

1. `README.md`
2. `scripts/HiDe/Train_*/*.sh`
3. `llava/train/train_MOE.py`
4. `HiDe/peft/tuners/clitmoelora.py`
5. `llava/model/llava_arch.py`
6. `llava/model/language_model/llava_llama.py`
7. `llava/model/builder.py`
8. `llava/eval/model_answer.py` and the metric scripts

---

## 3. The real training pipeline

The shell scripts are the clearest source of truth for how the project is meant to run.

### UCIT continual schedule

Under `scripts/HiDe/Train_UCIT/`:

- `Task1.sh` trains the first task from the base LLaVA checkpoint
- `Task2.sh` onward load `--previous_task_model_path`
- Each task increments `--cur_task`
- All tasks use `llava/train/train_mem_MOE.py`
- All tasks enable LoRA and pass `--expert_num 6`

The UCIT task order is:

1. ImageNet-R
2. ArxivQA
3. VizWiz-caption
4. IconQA
5. CLEVR-Math
6. Flickr30k-caption

So the continual behavior is not abstract only in code; it is concretely driven by the task scripts.

### CoIN schedule

Under `scripts/HiDe/Train_CoIN/`:

- Tasks are more LLaVA/CoIN-like
- They mainly use `llava/train/train_mem.py`, not the HiDe MoE trainer for the shown later tasks
- This part looks inherited from previous baselines / experiments and is less cleanly aligned with the UCIT path

### Important conclusion

The **main HiDe implementation path** is:

`Train_*.sh` -> `llava/train/train_mem_MOE.py` -> `llava/train/train_MOE.py` -> `HiDe/peft` + `llava/model/llava_arch.py`

---

## 4. Training code architecture

## 4.1 `llava/train/train.py`

This is the mostly standard LLaVA supervised fine-tuning trainer:

- parses model/data/training arguments
- builds tokenizer and model
- initializes vision modules
- prepares dataset
- optionally applies standard LoRA
- launches a custom HuggingFace trainer

Main responsibilities:

- conversation formatting
- masking labels so only assistant tokens contribute to loss
- lazy image loading
- multimodal batching
- save logic for full checkpoints / projector / LoRA states

Key parts:

- `ModelArguments`: model/backbone/multimodal settings
- `DataArguments`: JSON data path, optional replay memory path, image folder
- `TrainingArguments`: HF args plus LoRA / quantization / grouping options
- `LazySupervisedDataset`: loads JSON samples lazily and reads images at access time
- `preprocess*()` functions: build prompt text and label masks

## 4.2 `llava/train/train_MOE.py`

This is the **HiDe-specific training entrypoint**.

It is structurally very close to `train.py`, but changes the PEFT path:

- imports from local `HiDe.peft`
- introduces extra args:
  - `text_tower`
  - `mm_text_select_layer`
  - `cur_task`
  - `task_embedding_dim`
  - `expert_num`
- when `lora_enable=True`, it builds `HiDeMOELoraConfig`
- uses `TaskType.CAUSAL_LM_HiDe`

So the training code itself is not where the method lives mathematically; it mainly wires the method into the model.

## 4.3 `llava/train/llava_trainer.py`

Custom trainer features:

- optional modality-aware batching
- special optimizer grouping for `mm_projector`
- Deepspeed-friendly parameter extraction

The most important custom behavior is `group_by_modality_length`, which groups multimodal and text-only lengths more efficiently for training.

---

## 5. Data flow during training

Each training sample is expected to look like LLaVA-format JSON:

- `conversations`: list of `human` / `gpt` turns
- optional `image`

Flow:

1. `LazySupervisedDataset` loads JSON records
2. Image is read from `image_folder`
3. `preprocess_multimodal()` injects `<image>` tokens
4. `preprocess_*()` builds a conversation prompt for the selected template
5. Human tokens are masked to `IGNORE_INDEX`
6. Collator pads text and stacks images
7. Model receives `input_ids`, `labels`, and `images`

The code supports several conversation styles:

- Vicuna-style
- LLaMA-2 chat style
- MPT style
- plain format

This is controlled by `conversation_lib.default_conversation`.

---

## 6. Model architecture

## 6.1 Base multimodal stack

The multimodal structure stays close to LLaVA:

- language model: `LlavaLlamaForCausalLM`
- vision encoder: CLIP vision model with projection output
- projector: vision features -> LM hidden size
- optional text tower: CLIP text encoder for guidance / routing

Relevant files:

- `llava/model/language_model/llava_llama.py`
- `llava/model/llava_arch.py`
- `llava/model/multimodal_encoder/clip_encoder.py`
- `llava/model/multimodal_projector/builder.py`

## 6.2 `llava/model/language_model/llava_llama.py`

This defines the main LLaVA language model wrapper.

HiDe-specific additions:

- `self.cur_task`
- `self.expert_num`
- `self.image_anchors`
- `self.text_anchors`
- `self.image_boundary`
- `self.text_boundary`

These anchor/boundary tensors are central to how HiDe tracks task-specific multimodal prototypes.

## 6.3 `llava/model/multimodal_encoder/clip_encoder.py`

Two frozen towers are defined:

- `CLIPVisionTower`
  - uses `CLIPVisionModelWithProjection`
  - returns both projected global image embedding and patch features
- `CLIPTextTower`
  - uses `CLIPTextModel`
  - returns pooled text embedding

This means HiDe uses:

- image global embedding for task similarity
- text pooled embedding for task similarity
- image patch tokens for actual LLaVA token insertion

## 6.4 `llava/model/multimodal_projector/builder.py`

The projector is standard LLaVA-style:

- `linear`
- `mlpNx_gelu`
- `identity`

The provided `config.json` sets:

- `mm_projector_type = "mlp2x_gelu"`

---

## 7. The most important file: `llava/model/llava_arch.py`

This file contains the **actual HiDe routing logic** inside the multimodal forward preparation.

### What upstream LLaVA already did here

Upstream responsibilities:

- build and initialize vision modules
- encode images
- replace `<image>` tokens with projected image patch embeddings
- align labels and attention masks

### What HiDe adds

HiDe adds a second guide pathway:

1. obtain global image embedding from CLIP vision projection
2. decode the current prompt back to text
3. convert prompt text into CLIP text inputs
4. obtain text guidance embedding from CLIP text tower
5. use image/text anchors to update task prototypes during training
6. use anchor similarity during inference to compute expert weights

### Training-time behavior

During training:

- current batch image features and text features are accumulated into task-specific anchors
- anchors are updated by running-average logic using `image_boundary` and `text_boundary`
- the active task is `self.cur_task`

So each task gradually builds:

- an image prototype
- a text prototype

### Inference-time behavior

During inference:

- current sample image/text features are compared against all stored anchors
- cosine similarities are computed
- image and text similarities are averaged
- softmax over tasks gives `compute_expert_weight`
- these weights are written into selected projection layers in the **last LM block**

Specifically, the code sets `expert_weight` only on:

- attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`
- only for `self.model.layers[-1]`

### Practical interpretation

The design is:

- earlier LoRA experts are cumulatively fused for most layers
- the final transformer block keeps per-expert weighted combination at inference
- routing is driven by multimodal task similarity

This is the main implementation of “hierarchical decoupling” in code form.

### Why this file matters most

If you want to change the method itself, start here first.

---

## 8. HiDe custom PEFT fork

The folder `HiDe/peft/` is a fork of Hugging Face PEFT with one extra adapter family.

### Standard fork contents

Most files are near-upstream PEFT:

- config objects
- model wrappers
- LoRA / AdaLoRA / IA3 / prompt tuning
- load/save helpers

### HiDe additions

The key extensions are:

- new `PeftType`: `MOE_LORA_HiDe`
- new `TaskType`: `CAUSAL_LM_HiDe`
- new config: `HiDeMOELoraConfig`
- new model wrapper: `HiDeMOELoraModel`
- new layer types in `tuners/clitmoelora.py`

## 8.1 `HiDe/peft/mapping.py`

This wires the new task type to a new model wrapper:

- `CAUSAL_LM_HiDe` -> `PeftModelForCausalLMLORAMOE`

## 8.2 `HiDe/peft/peft_model.py`

This is mostly PEFT infrastructure, but important because:

- it decides how adapters are wrapped around the base model
- it handles adapter save/load
- it exposes `PeftModel.from_pretrained()`

For HiDe, the special wrapper is `PeftModelForCausalLMLORAMOE`.

## 8.3 `HiDe/peft/tuners/clitmoelora.py`

This is the **second most important file** in the whole repository.

It defines:

- `HiDeMOELoraConfig`
- `HiDeMOELoraModel`
- `HiDeMOELoraLayer`
- `HiDeMOELoraLinear`
- `HiDeMOELinearA`
- `HiDeMOELinearB`
- `HiDeMOEExpert`

### How the adapter works

Each replaced linear layer gets:

- frozen base linear weight
- LoRA A/B blocks split across `expert_num` experts
- task-dependent behavior via `cur_task`
- inference-time weighted mixing via `expert_weight`

### Training behavior

In training:

- only the current task expert is used:
  - `loraA[cur_task]`
  - `loraB[cur_task]`

This gives task-specific expansion.

### Inference behavior

In inference:

- for most layers, experts up to `cur_task` are simply fused with equal weight
- for the **last transformer layer**, expert outputs are combined using `expert_weight`

That makes the last block the main task-adaptive fusion point.

### Important nuance

There are classes named `Gate` / `Router`, but the actually used routing in the LLaVA forward path is mostly the anchor-similarity logic from `llava_arch.py`, not a full learned token router.

So conceptually this repo is more:

- **task-prototype-guided MoE-LoRA**

than:

- fully general token-level sparse MoE

---

## 9. Model loading and checkpoint behavior

`llava/model/builder.py` is the inference load path.

It supports:

- base LLaVA models
- LoRA LLaVA models
- projector-only checkpoints
- plain language models

For HiDe-LLaVA LoRA checkpoints:

1. load base LLaVA model
2. load extra non-LoRA trainables from `non_lora_trainables.bin`
3. load local `HiDe.peft.PeftModel`
4. merge / unload LoRA weights for inference
5. load both vision and text towers

This file also attaches:

- main tokenizer
- CLIP tokenizer used by the text tower

That CLIP tokenizer attachment is necessary because `llava_arch.py` decodes prompt text and re-encodes it for CLIP text guidance.

---

## 10. Inference flow

CLI inference path:

- `llava/serve/cli.py`

Server path:

- `llava/serve/model_worker.py`
- `llava/serve/controller.py`
- `llava/serve/gradio_web_server.py`

### Runtime steps

1. load model through `llava/model/builder.py`
2. build prompt from conversation template
3. preprocess image with CLIP image processor
4. replace `<image>` token using `tokenizer_image_token()`
5. call `model.generate(...)`
6. inside forward, `prepare_inputs_labels_for_multimodal()` injects image patch embeddings and computes HiDe routing weights

So even plain generation goes through the HiDe routing logic automatically.

---

## 11. Evaluation code

There are two layers:

### Layer A: answer generation scripts

- `llava/eval/model_answer.py`
- `llava/eval/model_others.py`
- `llava/eval/model_science_qa.py`

These:

- load the trained checkpoint
- iterate over a dataset JSON
- build prompts
- run `model.generate`
- write JSONL predictions

`model_answer.py` is the UCIT path and explicitly passes `text_tower` to model loading.

### Layer B: metric scripts

Each dataset then has a simple scorer:

- `eval_imagenet.py`
- `eval_vizwiz.py`
- `eval_vqav2.py`
- `eval_ocrvqa.py`
- `eval_grounding.py`
- `eval_science_qa.py`
- `eval_textvqa.py`
- `eval_caption.py`
- `eval_deepseek_r1.py`
- `eval_gqa.py`

Most are lightweight exact-match or dataset-specific evaluators.

Two special cases:

- `eval_caption.py`: caption metrics and optional LLM-based judging
- `eval_deepseek_r1.py`: uses DeepSeek/OpenAI-style API judging for open-ended tasks like ArxivQA / CLEVR / IconQA

---

## 12. Script orchestration

The shell scripts under `scripts/HiDe/` are very important because they encode:

- task order
- dataset paths
- GPU layout
- which Python module to run
- how results are merged
- which metric script scores each dataset

### UCIT eval pattern

Typical pattern:

1. run `python -m llava.eval.model_answer`
2. write chunked JSONL files
3. merge JSONL files
4. run a task-specific evaluator

### What each UCIT eval uses

- ArxivQA -> `eval_deepseek_r1.py`
- CLEVR-Math -> `eval_deepseek_r1.py`
- IconQA -> `eval_deepseek_r1.py`
- ImageNet-R -> `eval_imagenet.py`
- VizWiz-caption -> `eval_deepseek_r1.py` or caption path depending on split
- Flickr30k-caption -> `eval_caption.py`

---

## 13. Conversation and multimodal token utilities

### `llava/conversation.py`

Defines:

- prompt templates
- separator styles
- image-aware message serialization
- Gradio rendering helpers

This file is why the same dataset can be packed differently for Vicuna, MPT, or LLaMA-2 prompt styles.

### `llava/mm_utils.py`

Defines:

- image padding helper
- image tensor preprocessing wrapper
- `<image>` token insertion into tokenized text
- stopping criteria for generation

`tokenizer_image_token()` is especially important because LLaVA uses a sentinel token index for image insertion before actual embeddings are swapped in.

---

## 14. CKA analysis script

`test_CKA_sim.py` is not part of the main runtime pipeline.

It is an analysis script that:

- hooks model activations
- computes linear / kernel CKA similarity
- likely supports the paper’s claim about different layer behaviors across tasks

This aligns with the paper motivation for hierarchical decoupling.

---

## 15. What is upstream vs. what is novel here

### Mostly upstream / inherited

- large parts of `llava/train/`
- large parts of `llava/serve/`
- most of `llava/model/` outside the HiDe changes
- most of `HiDe/peft/` outside the new config/types/tuner classes

### HiDe-specific core

- `llava/train/train_MOE.py`
- `llava/model/llava_arch.py` custom anchor/routing logic
- `llava/model/language_model/llava_llama.py` anchor storage / task fields
- `HiDe/peft/tuners/clitmoelora.py`
- `HiDe/peft/mapping.py`
- `HiDe/peft/utils/config.py` new enum values
- task shell scripts that define continual order

If you want to isolate the paper contribution, those are the highest-value files.

---

## 16. Important implementation ideas in plain language

HiDe-LLaVA’s code implements this rough idea:

1. Train tasks sequentially
2. Give each task its own LoRA expert slot
3. Keep multimodal prototypes for each task using:
   - global image CLIP embeddings
   - text CLIP embeddings of the prompt
4. At inference, compare the current sample to stored task prototypes
5. Use the similarity scores to decide how much each expert should contribute
6. Apply that fusion most explicitly in the last transformer block

So the “hierarchical decoupling” is reflected by:

- task-specific expansion during training
- task-general fusion during inference

---

## 17. Practical code-reading advice

If you want to modify behavior:

### Change task routing logic

Edit:

- `llava/model/llava_arch.py`

### Change expert architecture

Edit:

- `HiDe/peft/tuners/clitmoelora.py`

### Change task order / continual setup

Edit:

- `scripts/HiDe/Train_UCIT/*.sh`
- `scripts/HiDe/Train_CoIN/*.sh`

### Change prompt packing / masking

Edit:

- `llava/train/train.py`
- `llava/train/train_MOE.py`
- `llava/conversation.py`

### Change checkpoint loading

Edit:

- `llava/model/builder.py`

---

## 18. Repository quirks and caveats

There are a few signs this repo is research code rather than a productionized library:

- many hard-coded absolute paths in scripts
- `sys.path.append(...)` hacks in Python files
- script names and README references sometimes still say `CoIN`
- some eval imports appear inherited from another folder structure
- some files are near-duplicates of upstream LLaVA

So when debugging, assume:

- the shell scripts are closer to the authors’ real workflow than generic module interfaces
- UCIT path is the cleaner HiDe path
- CoIN path contains more legacy structure

---

## 19. Short summary

If you only remember five things:

1. `train_MOE.py` is the main HiDe training entrypoint.
2. `clitmoelora.py` defines the custom HiDe MoE-LoRA adapter.
3. `llava_arch.py` contains the anchor-based task routing logic.
4. `llava_llama.py` stores per-task anchors and state.
5. `scripts/HiDe/Train_UCIT/*.sh` define the actual continual learning schedule.

---

## 20. Suggested next reading order

For detailed comprehension, read next in this exact order:

1. `scripts/HiDe/Train_UCIT/Task1.sh`
2. `scripts/HiDe/Train_UCIT/Task2.sh`
3. `llava/train/train_MOE.py`
4. `HiDe/peft/tuners/clitmoelora.py`
5. `llava/model/llava_arch.py`
6. `llava/model/language_model/llava_llama.py`
7. `llava/model/builder.py`
8. `llava/eval/model_answer.py`

That sequence matches the real execution path of the method.
