path=/code_path/seq2seq_img_dualkd_mbart
img_lr_factor=$1
cross_attn_type=$2
fusion_layer=$3
use_logit_kd=$4
lambda_kd=$5
temperature=$6
echo $cross_attn_type, $use_logit_kd
if [[ "$use_logit_kd" = "True" ]]; then
    echo $cross_attn_type, $use_logit_kd, "yes"
    bash $path/multimodal_trainer_mbart50.sh --ngpus 8 --training_type m2m --img_lr_factor ${img_lr_factor} --cross_attn_type ${cross_attn_type} --fusion_layer ${fusion_layer} --use_logit_kd --lambda_kd ${lambda_kd} --temperature ${temperature}
else
    echo $cross_attn_type, $use_logit_kd, "no"
    bash $path/multimodal_trainer_mbart50.sh --ngpus 8 --training_type m2m --img_lr_factor ${img_lr_factor} --cross_attn_type ${cross_attn_type} --fusion_layer ${fusion_layer} --lambda_kd ${lambda_kd} --temperature ${temperature}
fi
# trains the many-to-many model
#bash $path/trainer.sh --ngpus 8 --training_type m2o --pivot_lang arabic # trains the many-to-one model using arabic as the target language
#bash $path/trainer.sh --ngpus 8 --training_type o2m --pivot_lang english # trains the one-to-many model using english as the source language
