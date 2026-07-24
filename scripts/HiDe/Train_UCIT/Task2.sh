################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

launch_ucit_task "$UCIT_TASK2_TRAIN_JSON" "$UCIT_OUTPUT_ROOT/Task2_llava_lora_ours" 1 "$UCIT_OUTPUT_ROOT/Task1_llava_lora_ours"
