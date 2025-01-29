set -ex

#MODEL_NAME_OR_PATH=/pubshare/zy/cache/Qwen2.5-Math-1.5B-Instruct
MODEL_NAME_OR_PATH=$1
PROMPT_TYPE="pure"
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eva_newdataset

SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
# DATA_NAME="gsm8k,math,svamp,mmlu_stem,aqua"
# DATA_NAME="gsm8k,math,minerva_math,gaokao2023en,olympiadbench,college_math,mmlu_stem"
DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math,aqua,sat_math,mmlu_stem,aime24,amc23"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
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

