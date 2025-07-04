# Script for the model fine-tuned based on GRT-SFT strategy, which calculates the accuracy in predicting problems in the picture.
# only related to the POPE task of VoQA dataset.
# 1. The prompt id needs to correspond to the prompt used in reasoning. For GRT-SFT, default is 0
PROMPT_ID=0
# 2. VoQA dataset folder names (concatenation or watermark rendering)
METHOD_FOLDER='watermark_rendering'
# 3. The model name, which should correspond to the saved json or jsonl.
MODEL_NAME="llava-v1.5-7b-GRT"
# 4. Concatenation directions (for watermark rendering, just keep 'no')
DIRECTION='no'
# 5. VoQA dataset path (root path)
EVAL_DIR="path/to/VoQA/evaluation/dataset"
# 6. Whether the answers will be further carefully filtered (for zero-shot models, default is true; for training models, default is false)
FILTER_ANSWER="false"
# 7. Trigger token in the models, which is case-insensitive. Default is 'ASSISTANT'.
SPLIT_WORD="ASSISTANT"
# 8. The type of the model relative to the VoQA dataset. Default is 'grt-sft'.
MODEL_TYPE="grt-sft"
# 9. The tasks in VoQA, you can choose in [scienceqa, textvqa, pope, gqa]
TASKS=(scienceqa textvqa pope gqa)
# 10. Special args for GQA only
GQA_SPLIT="llava_gqa_testdev_balanced"
GQA_TIER="testdev_balanced" # tier in GQA scripts


for TASK in "${TASKS[@]}"; do

    if [ "$TASK" = "scienceqa" ]; then
        echo "Current task: $TASK"
        python scripts_for_result_analysis/eval_others.py \
            --task $TASK \
            --sqa_resoning_json $EVAL_DIR/scienceqa/answers_prompt$PROMPT_ID/scienceqa_${METHOD_FOLDER}_image/${MODEL_NAME}_${DIRECTION}.jsonl \
            --sqa_prediction_json $EVAL_DIR/scienceqa/answers_prompt$PROMPT_ID/scienceqa_${METHOD_FOLDER}_image/${MODEL_NAME}_${DIRECTION}_output.jsonl \
            --sqa_save_file $EVAL_DIR/scienceqa/answers_prompt$PROMPT_ID/scienceqa_${METHOD_FOLDER}_image/${MODEL_NAME}_${DIRECTION}_for_analysis.jsonl
    elif [ "$TASK" = "textvqa" ]; then
        echo "Current task: $TASK"
        # step 1: get id_to_score json
        python scripts_for_result_analysis/eval_textvqa.py \
            --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val_new_id.json \
            --result-file $EVAL_DIR/textvqa/answers_prompt${PROMPT_ID}/textvqa_${METHOD_FOLDER}_image/${MODEL_NAME}_${DIRECTION}.jsonl \
            --filter_answer $FILTER_ANSWER \
            --split_word $SPLIT_WORD \
            --model_type $MODEL_TYPE \
            --textvqa_id_to_score_json $EVAL_DIR/textvqa/answers_prompt$PROMPT_ID/textvqa_${METHOD_FOLDER}_image/${MODEL_NAME}_${DIRECTION}_score_to_qids.json 
        # step 2: calculate score
        python scripts_for_result_analysis/eval_others.py \
            --task $TASK \
            --textvqa_id_to_score_json $EVAL_DIR/textvqa/answers_prompt$PROMPT_ID/textvqa_${METHOD_FOLDER}_image/${MODEL_NAME}_${DIRECTION}_score_to_qids.json \
            --textvqa_prediction_json $EVAL_DIR/textvqa/answers_prompt$PROMPT_ID/textvqa_${METHOD_FOLDER}_image/${MODEL_NAME}_${DIRECTION}.jsonl \
            --textvqa_save_file $EVAL_DIR/textvqa/answers_prompt$PROMPT_ID/textvqa_${METHOD_FOLDER}_image/${MODEL_NAME}_${DIRECTION}_for_analysis.jsonl
    elif [ "$TASK" = "pope" ]; then
        echo "Current task: $TASK"
        python scripts_for_result_analysis/eval_pope.py \
            --annotation-dir $EVAL_DIR/pope/coco \
            --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
            --result-file $EVAL_DIR/pope/answers_prompt${PROMPT_ID}/pope_${METHOD_FOLDER}_image/${MODEL_NAME}_${DIRECTION}.jsonl \
            --filter_answer $FILTER_ANSWER \
            --split_word $SPLIT_WORD \
            --model_type $MODEL_TYPE
    elif [ "$TASK" = "gqa" ]; then
        echo "Current task: $TASK"
        python scripts_for_result_analysis/eval_others.py \
            --task $TASK \
            --gqa_reference_json $EVAL_DIR/gqa/${GQA_TIER}_questions.json \
            --gqa_reasoning_json $EVAL_DIR/gqa/answers/${GQA_SPLIT}_prompt${PROMPT_ID}/${MODEL_NAME}_${DIRECTION}/gqa_${METHOD_FOLDER}_image/merge.jsonl \
            --gqa_prediction_json $EVAL_DIR/gqa/answers/${GQA_SPLIT}_prompt${PROMPT_ID}/${MODEL_NAME}_${DIRECTION}/gqa_${METHOD_FOLDER}_image/${GQA_TIER}_predictions.json    
    else
        echo "Unknown task name: $TASK. exit!"
    fi
    
done
