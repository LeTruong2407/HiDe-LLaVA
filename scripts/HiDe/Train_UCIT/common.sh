#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../paths.sh"

setup_train_launcher() {
  TRAIN_CMD=()
  TRAIN_EXTRA_PREFIX=("")

  case "$TRAIN_LAUNCHER" in
    auto)
      if command -v deepspeed >/dev/null 2>&1; then
        TRAIN_CMD=(deepspeed --include "$DEEPSPEED_INCLUDE" --master_port "$MASTER_PORT" llava/train/train_mem_MOE.py)
        TRAIN_EXTRA_PREFIX=(--deepspeed "$DEEPSPEED_CONFIG")
      else
        TRAIN_CMD=(python -m llava.train.train_mem_MOE)
      fi
      ;;
    deepspeed)
      TRAIN_CMD=(deepspeed --include "$DEEPSPEED_INCLUDE" --master_port "$MASTER_PORT" llava/train/train_mem_MOE.py)
      TRAIN_EXTRA_PREFIX=(--deepspeed "$DEEPSPEED_CONFIG")
      ;;
    python)
      TRAIN_CMD=(python -m llava.train.train_mem_MOE)
      ;;
    *)
      echo "Unsupported TRAIN_LAUNCHER='$TRAIN_LAUNCHER'. Use auto, deepspeed, or python."
      exit 1
      ;;
  esac
}

launch_ucit_task() {
  local task_json="$1"
  local task_output="$2"
  local cur_task="$3"
  local previous_task_model_path="${4:-}"

  setup_train_launcher

  local cmd=("${TRAIN_CMD[@]}")
  if [ "${#TRAIN_EXTRA_PREFIX[@]}" -gt 0 ] && [ -n "${TRAIN_EXTRA_PREFIX[0]}" ]; then
    cmd+=("${TRAIN_EXTRA_PREFIX[@]}")
  fi
  cmd+=(
    --lora_enable True --lora_r 48 --lora_alpha 96 --mm_projector_lr 2e-5
    --expert_num 6
    --model_name_or_path "$LLAVA_BASE_MODEL"
    --version "$PROMPT_VERSION"
    --data_path "$task_json"
    --image_folder "$DATA_ROOT"
    --vision_tower "$CLIP_MODEL"
    --text_tower "$CLIP_MODEL"
    --cur_task "$cur_task"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 "$TRAIN_BF16"
    --output_dir "$task_output"
    --num_train_epochs "$TRAIN_NUM_EPOCHS"
    --per_device_train_batch_size "$TRAIN_PER_DEVICE_BATCH"
    --per_device_eval_batch_size "$TRAIN_PER_DEVICE_EVAL_BATCH"
    --gradient_accumulation_steps "$TRAIN_GRAD_ACCUM_STEPS"
    --evaluation_strategy no
    --save_strategy "$TRAIN_SAVE_STRATEGY"
    --learning_rate 2e-4
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps "$TRAIN_LOGGING_STEPS"
    --tf32 "$TRAIN_TF32"
    --model_max_length "$TRAIN_MODEL_MAX_LENGTH"
    --gradient_checkpointing True
    --dataloader_num_workers "$TRAIN_NUM_WORKERS"
    --lazy_preprocess True
    --report_to none
  )

  if [ -n "$previous_task_model_path" ]; then
    cmd+=(--previous_task_model_path "$previous_task_model_path")
  else
    cmd+=(--pretrain_mm_mlp_adapter "$LLAVA_BASE_MODEL/mm_projector.bin")
  fi

  if [ -n "$EXTRA_TRAIN_ARGS" ]; then
    # shellcheck disable=SC2206
    local extra_args=( $EXTRA_TRAIN_ARGS )
    cmd+=("${extra_args[@]}")
  fi

  echo "Launching training with: ${cmd[*]}"
  "${cmd[@]}"
}
