#!/bin/bash
export HDF5_USE_FILE_LOCKING=false
ARGPARSE_DESCRIPTION="Trainer utility"
source $(dirname $0)/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1

parser.add_argument('--ngpus', default=8, type=int,
                    help='No. of gpus to use')
parser.add_argument('--training_type', type=str, choices=["m2m", "m2o", "o2m"],
                    required=True, help='Training type (many-to-many/many-to-one/one-to-many)')
parser.add_argument('--pivot_lang', type=str, default="english",
                    help='Pivot language (Applicable for many-to-one and one-to-many)')
parser.add_argument('--exclude_native', action='store_true',
                    default=False, help='Exclude the native-to-native filepairs during training')
parser.add_argument('--cross_attn_type', default=4, type=int,
                    help='No. of cross_attn_type to use')
parser.add_argument('--fusion_layer', default=11, type=int,
                    help='No. of fusion_layer to use')
parser.add_argument('--max_img_len', default=108, type=int,
                    help='No. of fusion_layer to use')
parser.add_argument('--img_lr_factor', default=str, type=float,
                    help='No. of cross_attn_type to use')

parser.add_argument('--use_logit_kd', action='store_true',
                    help='No. of cross_attn_type to use')
parser.add_argument('--lambda_kd', default=0.5, type=float,
                    help='No. of cross_attn_type to use')
parser.add_argument('--temperature', default=2, type=float,
                    help='No. of cross_attn_type to use')
EOF
#echo "start"$cross_attn_type $img_lr_factor $ngpus
#echo "TRAINING_TYPE"$TRAINING_TYPE
#echo "PIVOT_LANG"$PIVOT_LANG
export PREFIX="${TRAINING_TYPE}_${PIVOT_LANG}"
export BASE_DIR=$(realpath .)
export BASE_DIR=/data_path
export ROOT_DATASET_DIR="${BASE_DIR}/dataset"
export ROOT_INPUT_DIR="${BASE_DIR}/${TRAINING_TYPE}"
sign=output_model_with_img_dualkd_nosharedvanillaKD_use_logit_kd${USE_LOGIT_KD}_lambda_kd${LAMBDA_KD}_temperature${TEMPERATURE}_img_lr_factor${IMG_LR_FACTOR}_mbart50
#sign=output_model_with_img_dualkd_from_teacher_use_logit_kd${USE_LOGIT_KD}_lambda_kd${LAMBDA_KD}_temperature${TEMPERATURE}
export ROOT_OUTPUT_DIR="${BASE_DIR}/${sign}"

if [ ! -d $ROOT_OUTPUT_DIR ]; then
    mkdir $ROOT_OUTPUT_DIR
    chmod 777 $ROOT_OUTPUT_DIR -R
fi

#export PREFIX="${TRAINING_TYPE}_${PIVOT_LANG}"
if [[ "$TRAINING_TYPE" = "m2m" ]]; then
    PREFIX="${TRAINING_TYPE}"
    OPTIONAL_ARGS=(
        "--multistage_upsampling_factors 0.5 0.75"
    )
    if [[ "$SAMPLING" = "unistage" ]]; then
        OPTIONAL_ARGS=(
            "--upsampling_factor 0.25"
        )   
    fi
else
    OPTIONAL_ARGS=(
        "--upsampling_factor 0.75"
    )
fi

export SUFFIX="with_native"
if [[ "$EXCLUDE_NATIVE" = "yes" ]]; then
    SUFFIX="without_native"
fi

export BASENAME="${PREFIX}_${SUFFIX}"
export INPUT_DIR="${ROOT_INPUT_DIR}"
export OUTPUT_DIR="${ROOT_OUTPUT_DIR}/${BASENAME}"
export MIN_EXAMPLE_COUNT=32

if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
    chmod 777 $OUTPUT_DIR -R
fi

#conda activate "${BASE_DIR}/env" || source activate "${BASE_DIR}/env"
if false; then
if [[ "${SLURM_PROCID:-0}" -eq 0 && "${SLURM_LOCALID:-0}" -eq 0 ]]; then
    mkdir -p $OUTPUT_DIR
    python "${BASE_DIR}/generate_data.py" \
        --dataset_dir $ROOT_DATASET_DIR \
        --output_dir $INPUT_DIR \
        --training_type $TRAINING_TYPE \
        --pivot_lang $PIVOT_LANG \
        --exclude_native $EXCLUDE_NATIVE \
        --min_example_count $MIN_EXAMPLE_COUNT
fi
fi
# for ozstar only; the model must
# be cached if this variable is set
export LINK_CACHE_ONLY=false 

# training settings
export max_steps=10000
export save_steps=2000
export logging_steps=100

# validation settings
export evaluation_strategy="no"

# model settings
#export model_name="google/mt5-base"
export model_name="/model_to/mbart-large-50-many-to-many-mmt/"
# optimization settings
export learning_rate=1
export warmup_steps=2000
export gradient_accumulation_steps=10
export weight_decay=0.01
export lr_scheduler_type="transformer"
#export lr_scheduler_type="linear"
export label_smoothing_factor=0.1

# misc. settings
export seed=1234

# input / output settings
export input_dir=$INPUT_DIR
export output_dir=$OUTPUT_DIR

# batch / sequence sizes
export PER_DEVICE_TRAIN_BATCH_SIZE=3
export MAX_SOURCE_LENGTH=512
export MAX_TARGET_LENGTH=84

# cross lingual settings
export per_lang_batch_size=30

# logging settings
export WANDB_PROJECT="Crossum"
export WANDB_WATCH=false
export WANDB_DISABLED=true
export HDF5_USE_FILE_LOCKING=false
export HDF5_USE_FILE_LOCKING=FALSE
BASE_DIR=/code_path/seq2seq_img_dualkd_mt5
export HOME=/root_path
NPROC_PER_NODE=$NGPUS
#INPUT_DIR=
#python -m torch.distributed.launch \
#		--nproc_per_node=${NPROC_PER_NODE:-$NGPUS} \
#		--nnodes=${SLURM_JOB_NUM_NODES:-1} \
#		--node_rank=${SLURM_PROCID:-0} \
#		--master_addr="${PARENT:-127.0.0.1}" --master_port="${MPORT:-29500}" "${BASE_DIR}/pipeline.py" \
if [[ "$USE_LOGIT_KD" == "yes" ]]; then
    echo "use_logit_kd", $USE_LOGIT_KD
#    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
#                   --nproc_per_node=$NPROC_PER_NODE \
#                    "${BASE_DIR}/pipeline.py" \
     CUDA_VISIBLE_DEVICES=0 python "${BASE_DIR}/pipeline.py" \
        --model_name_or_path $model_name \
        --data_dir $INPUT_DIR --output_dir $OUTPUT_DIR \
        --learning_rate=$learning_rate --warmup_steps $warmup_steps --gradient_accumulation_steps $gradient_accumulation_steps \
        --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type --adafactor --label_smoothing_factor $label_smoothing_factor \
        --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --logging_steps $logging_steps \
        --max_source_length $MAX_SOURCE_LENGTH --max_target_length $MAX_TARGET_LENGTH \
        --per_lang_batch_size $per_lang_batch_size \
        --seed $seed --overwrite_output_dir \
        --max_steps $max_steps --save_steps $save_steps \
        --evaluation_strategy $evaluation_strategy  \
        --logging_first_step \
        --cache_dir "${BASE_DIR}/cache_dir" \
        --run_name $BASENAME \
        --use_langid \
        --langid_map_path "${BASE_DIR}/debug/extra_tokens_langid_map.json" \
        --reinitialize_langid_embeddings "bos" \
        --do_train \
        --use_logit_kd \
        --lambda_kd ${LAMBDA_KD} \
        --temperature ${TEMPERATURE} \
        --max_img_len=${MAX_IMG_LEN} \
        --img_lr_factor=${IMG_LR_FACTOR} \
        --use_forget_gate \
        --cross_attn_type=${CROSS_ATTN_TYPE} \
        --use_img_trans \
        --n_attn_heads=8 \
        --fusion_layer=${FUSION_LAYER} \
        $(echo -n ${OPTIONAL_ARGS[@]}) |& tee "${OUTPUT_DIR}/run.log"
else
    echo "no use_logit_kd", $USE_LOGIT_KD
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
                   --nproc_per_node=$NPROC_PER_NODE \
                    "${BASE_DIR}/pipeline.py" \
        --model_name_or_path $model_name \
        --data_dir $INPUT_DIR --output_dir $OUTPUT_DIR \
        --learning_rate=$learning_rate --warmup_steps $warmup_steps --gradient_accumulation_steps $gradient_accumulation_steps \
        --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type --adafactor --label_smoothing_factor $label_smoothing_factor \
        --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --logging_steps $logging_steps \
        --max_source_length $MAX_SOURCE_LENGTH --max_target_length $MAX_TARGET_LENGTH \
        --per_lang_batch_size $per_lang_batch_size \
        --seed $seed --overwrite_output_dir \
        --max_steps $max_steps --save_steps $save_steps \
        --evaluation_strategy $evaluation_strategy  \
        --logging_first_step \
        --cache_dir "${BASE_DIR}/cache_dir" \
        --run_name $BASENAME \
        --use_langid \
        --langid_map_path "${BASE_DIR}/debug/extra_tokens_langid_map.json" \
        --reinitialize_langid_embeddings "bos" \
        --do_train \
        --lambda_kd ${LAMBDA_KD} \
        --temperature ${TEMPERATURE} \
        --max_img_len=${MAX_IMG_LEN} \
        --img_lr_factor=${IMG_LR_FACTOR} \
        --use_forget_gate \
        --cross_attn_type=${CROSS_ATTN_TYPE} \
        --use_img_trans \
        --n_attn_heads=8 \
        --fusion_layer=${FUSION_LAYER} \
        $(echo -n ${OPTIONAL_ARGS[@]}) |& tee "${OUTPUT_DIR}/run.log"
fi
