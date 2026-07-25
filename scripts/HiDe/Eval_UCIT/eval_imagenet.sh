#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../paths.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-32}"
EVAL_QUANT_ARGS="${EVAL_QUANT_ARGS:---load-4bit}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"

gpu_list="${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS="${EVAL_CHUNKS:-${#GPULIST[@]}}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 STAGE MODELPATH [GPU]"
    echo "Example: $0 consensus-task1 outputs/ucit_consensus/Task1_llava_lora_ours 0"
    exit 1
fi

STAGE="$1"
MODELPATH="$2"
GPU_ARG="${3:-}"
if [ -n "$GPU_ARG" ]; then
    GPULIST=("$GPU_ARG")
    CHUNKS=1
fi

RESULT_DIR="${RESULT_DIR:-./results/UCIT/each_dataset/ImageNet-R}"
mkdir -p "$RESULT_DIR/$STAGE"

EVAL_SAMPLE_ARGS=()
if [ -n "$EVAL_MAX_SAMPLES" ]; then
    EVAL_SAMPLE_ARGS=(--max-samples "$EVAL_MAX_SAMPLES")
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_INDEX=$((IDX % ${#GPULIST[@]}))
    GPU="${GPULIST[$GPU_INDEX]}"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -m llava.eval.model_answer \
        --model-path "$MODELPATH" \
        --model-base "$LLAVA_BASE_MODEL" \
        --question-file "$INSTRUCTION_ROOT/ImageNet-R/test_3000.json" \
        --image-folder "$DATA_ROOT" \
        --text-tower "$CLIP_MODEL" \
        --answers-file "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
        --conv-mode vicuna_v1 \
        "${EVAL_SAMPLE_ARGS[@]}" \
        $EVAL_QUANT_ARGS &
done

wait
output_file="$RESULT_DIR/$STAGE/merge.jsonl"
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

"$PYTHON_BIN" -m llava.eval.eval_deepseek_r1 \
    --annotation-file "$INSTRUCTION_ROOT/ImageNet-R/test_3000.json" \
    --result-file "$output_file" \
    --output-dir "$RESULT_DIR/$STAGE"
