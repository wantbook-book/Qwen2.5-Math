#!/bin/bash
set -ex

# PROMPT_TYPE=$1
PROMPT_TYPE="direct"
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
# DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"
DATA_NAME="gsm8k,math,svamp,mmlu_stem,aqua"
MODEL_NAME_OR_PATH="/home/jovyan/share/LLMAgent/model/Llama-3.2-1B-Instruct"
CHECKPOINT_DIR_ROOT="/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-1b-porm_grpo_math_instruct/checkpoints/_actor"
CHECKPOINT_DIR=("${CHECKPOINT_DIR_ROOT}/global_step600_pretrain" "${CHECKPOINT_DIR_ROOT}/global_step700_pretrain" "${CHECKPOINT_DIR_ROOT}/global_step800_pretrain")  # 这里是你的检查点路径列表
MODEL_PATHS=("${CHECKPOINT_DIR_ROOT}/global_step600_pretrain" "${CHECKPOINT_DIR_ROOT}/global_step700_pretrain" "${CHECKPOINT_DIR_ROOT}/global_step800_pretrain")  # 这里是你的检查点路径列表
export TOKENIZERS_PARALLELISM=false
for model_path in "${MODEL_PATHS[@]}"; do
    CUDA_VISIBLE_DEVICES=6,7 python3 -u math_eval.py \
        --model_name_or_path ${model_path} \
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
        --checkpoint_suffix
done


# English multiple-choice datasets
# DATA_NAME="aqua,sat_math,mmlu_stem"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --num_shots 5

