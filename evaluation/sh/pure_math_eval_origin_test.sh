set -ex

#MODEL_NAME_OR_PATH=/pubshare/zy/cache/Qwen2.5-Math-1.5B-Instruct
MODEL_NAME_OR_PATH=$1
PROMPT_TYPE="pure"
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="mmlu_stem,aqua,math"
CUDA_VISIBLE_DEVICES=6 \
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval_test.py \
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
    --end 2 \
    --use_vllm \
    --save_outputs \
    --overwrite

