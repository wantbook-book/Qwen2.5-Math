#!/bin/bash
set -ex

# PROMPT_TYPE=$1
PROMPT_TYPE="pure"
# MODEL_NAME_OR_PATH=$1


SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
# DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"
# DATA_NAME="gsm8k,math,svamp,mmlu_stem,aqua"
# DATA_NAME="gsm8k,math,minerva_math,gaokao2023en,olympiadbench,college_math,mmlu_stem"
DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math,aqua,sat_math,mmlu_stem,aime24,amc23"
DATA_NAME="olympiadbench,college_math,aqua,sat_math,mmlu_stem,aime24,amc23"
MODEL_NAME_OR_PATH="/home/jovyan/share/LLMAgent/model/Llama-3.2-1B-Instruct"
# CHECKPOINT_DIR_ROOT="/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-1b-porm_grpo_math_instruct/checkpoints/_actor"
# CHECKPOINT_DIR_ROOT="/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-1b-prm/_actor"
# CHECKPOINT_DIR_ROOT="/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-1b-porm_old/_actor"
# CHECKPOINT_DIR_ROOT="/pubshare/fwk/orlhf_checkpoints/select_checkpoint/llama3-1b-porm_eos/_actor"
# CHECKPOINT_DIR_ROOT="/pubshare/fwk/orlhf_checkpoints/select_checkpoint/llama3-1b-porm_new/_actor"
# CHECKPOINT_DIR_ROOT="/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-1b-mcts_single_st_mixed_gt/_actor"
CHECKPOINT_DIR_ROOT="/pubshare/fwk/orlhf_checkpoints/checkpoint/llama3-1b-porm_118new/_actor"

# CHECKPOINT_DIR=("${CHECKPOINT_DIR_ROOT}/global_step600_pretrain" "${CHECKPOINT_DIR_ROOT}/global_step700_pretrain" "${CHECKPOINT_DIR_ROOT}/global_step800_pretrain")  # 这里是你的检查点路径列表
# MODEL_PATHS=("${CHECKPOINT_DIR_ROOT}/global_step600_pretrain" "${CHECKPOINT_DIR_ROOT}/global_step700_pretrain" "${CHECKPOINT_DIR_ROOT}/global_step800_pretrain")  # 这里是你的检查点路径列表
CKPT_NAMES=("global_step100_pretrain" "global_step200_pretrain" "global_step300_pretrain" "global_step400_pretrain")
# CKPT_NAMES=("global_step150_pretrain")

OUTPUT_DIR=${CHECKPOINT_DIR_ROOT}/math_eval_newdataset
export TOKENIZERS_PARALLELISM=false
for ckpt_name in "${CKPT_NAMES[@]}"; do
    CUDA_VISIBLE_DEVICES=3 python3 -u math_eval.py \
        --model_name_or_path "${CHECKPOINT_DIR_ROOT}/${ckpt_name}" \
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

