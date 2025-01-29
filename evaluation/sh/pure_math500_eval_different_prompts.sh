set -ex

#MODEL_NAME_OR_PATH=/pubshare/zy/cache/Qwen2.5-Math-1.5B-Instruct
MODEL_NAME_OR_PATH=$1
PROMPT_TYPE="pure"
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eva_newdataset

SPLIT="math500"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="math"
PROMPTS_FILES_DIR="/pubshare/fwk/code/MCGEP/mcts/prompts/MATH_QUESTION_PROMPTS"
PROMPTS_FILES=($(ls ${PROMPTS_FILES_DIR} ))
export TOKENIZERS_PARALLELISM=false
for prompt_file in "${PROMPTS_FILES[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u math_eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${DATA_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --overwrite \
        --prompt_file ${PROMPTS_FILES_DIR}/${prompt_file}
done

