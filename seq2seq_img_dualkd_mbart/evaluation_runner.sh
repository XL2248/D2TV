#!/bin/bash
export BASE_DIR=/root_path
ROOT_DATASET_DIR="${BASE_DIR}/m2m"
sign=$1 #model name
export ROOT_OUTPUT_DIR="${BASE_DIR}/${sign}"
ROOT_MODEL_DIR="${BASE_DIR}/${sign}"
RESULTS_DIR="${BASE_DIR}/${sign}"

code=/code_path/seq2seq_img_dualkd_mbart
export HOME=$RESULTS_DIR
for model_dir in $ROOT_MODEL_DIR/*/; do
    echo $model_dir
    suffix=$(basename $model_dir)
    read training_type pivot_lang rest <<< $(IFS="_"; echo $suffix)

    if [[ "$training_type" = "m2o" ]]; then
        required_str="--required_tgt_lang ${pivot_lang}"
    elif [[ "$training_type" = "o2m" ]]; then
        required_str="--required_src_lang ${pivot_lang}"
    else
        required_str=" "
    fi

    for data_type in "test"; do
        python $code/evaluator.py \
            --dataset_dir "${ROOT_DATASET_DIR}" \
            --output_dir "${RESULTS_DIR}/${suffix}" \
            --evaluation_type xlingual \
            --data_type ${data_type} \
            --xlingual_summarization_model_name_or_path $model_dir \
            $required_str
    done
done
