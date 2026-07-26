#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../paths.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8}"
EVAL_MIN_NEW_TOKENS="${EVAL_MIN_NEW_TOKENS:-1}"
EVAL_QUANT_ARGS="${EVAL_QUANT_ARGS---load-4bit}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
EVAL_FORCE_EXPERT="${EVAL_FORCE_EXPERT:-}"
EVAL_ISOLATE_EXPERT="${EVAL_ISOLATE_EXPERT:-0}"
EVAL_ADAPTIVE_ALL_LAYER_ROUTING="${EVAL_ADAPTIVE_ALL_LAYER_ROUTING:-0}"
EVAL_DEBUG_GENERATION="${EVAL_DEBUG_GENERATION:-0}"

gpu_list="${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS="${EVAL_CHUNKS:-${#GPULIST[@]}}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 STAGE MODELPATH [GPU]"
    echo "Example: $0 consensus-task outputs/ucit_consensus/Task1_llava_lora_ours 0"
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
QUESTION_FILE="${QUESTION_FILE:-$INSTRUCTION_ROOT/ImageNet-R/test_3000.json}"
ANNOTATION_FILE="${ANNOTATION_FILE:-$INSTRUCTION_ROOT/ImageNet-R/test_3000.json}"
mkdir -p "$RESULT_DIR/$STAGE"

EVAL_EXTRA_ARGS=()
EVAL_SCORER_ARGS=()
if [ -n "$EVAL_MAX_SAMPLES" ]; then
    EVAL_EXTRA_ARGS+=(--max-samples "$EVAL_MAX_SAMPLES")
    EVAL_SCORER_ARGS+=(--max-samples "$EVAL_MAX_SAMPLES")
fi
if [ -n "$EVAL_FORCE_EXPERT" ]; then
    EVAL_EXTRA_ARGS+=(--force-expert "$EVAL_FORCE_EXPERT")
fi
if [ "$EVAL_ISOLATE_EXPERT" = "1" ]; then
    EVAL_EXTRA_ARGS+=(--isolate-expert)
fi
if [ "$EVAL_ADAPTIVE_ALL_LAYER_ROUTING" = "1" ]; then
    EVAL_EXTRA_ARGS+=(--adaptive-all-layer-routing)
fi
if [ "$EVAL_DEBUG_GENERATION" = "1" ]; then
    EVAL_EXTRA_ARGS+=(--debug-generation)
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_INDEX=$((IDX % ${#GPULIST[@]}))
    GPU="${GPULIST[$GPU_INDEX]}"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -m llava.eval.model_answer \
        --model-path "$MODELPATH" \
        --model-base "$LLAVA_BASE_MODEL" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$DATA_ROOT" \
        --text-tower "$CLIP_MODEL" \
        --answers-file "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
        --min-new-tokens "$EVAL_MIN_NEW_TOKENS" \
        --conv-mode vicuna_v1 \
        "${EVAL_EXTRA_ARGS[@]}" \
        $EVAL_QUANT_ARGS &
done

wait

output_file="$RESULT_DIR/$STAGE/merge.jsonl"
tmp_output_file="$output_file.tmp"
rm -f "$tmp_output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" >> "$tmp_output_file"
done

mv -f "$tmp_output_file" "$output_file"

"$PYTHON_BIN" -m llava.eval.eval_imagenet \
    --test-file "$ANNOTATION_FILE" \
    --result-file "$output_file" \
    --output-dir "$RESULT_DIR/$STAGE" \
    "${EVAL_SCORER_ARGS[@]}"
