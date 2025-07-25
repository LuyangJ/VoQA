#!/bin/bash
MODEL_PATH="$1"
METHOD_FOLDER="$2"
DIRECTION="$3"
PROMPT="$4"
PROMPT_ID="$5"
EVAL_DIR="$6"
FILTER_ANSWER="$7"
SPLIT_WORD="$8"
MODEL_TYPE="$9"

echo "$MODEL_PATH $METHOD_FOLDER $TASK $DIRECTION $EVAL_DIR $FILTER_ANSWER $SPLIT_WORD $MODEL_TYPE"
echo "prompt $PROMPT prompt_id $PROMPT_ID Start!"

MODEL_NAME=$(basename ${MODEL_PATH})

python eval/eval_main.py \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A_selected_mm.jsonl \
    --image-folder $EVAL_DIR/scienceqa/$METHOD_FOLDER \
    --answers-file $EVAL_DIR/scienceqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}_${DIRECTION}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --direction $DIRECTION \
    --model-name $MODEL_NAME \
    --task scienceqa \
    --batch-size 1 \
    --prompt "$PROMPT" \
    --original_benchmark "false" 

python eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}_${DIRECTION}.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}_${DIRECTION}_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers_prompt${PROMPT_ID}/$METHOD_FOLDER/${MODEL_NAME}_${DIRECTION}_result.json \
    --filter_answer $FILTER_ANSWER \
    --split_word $SPLIT_WORD \
    --model_type $MODEL_TYPE