DATA=data/R2G_hr_dataset_processed_v1
SPLIT=test 

NAME=cc5k_r2g_${SPLIT}_preds
SAVE_DIR=cross_eval_outputs/${NAME}
CKPT=hf:cubicasa5k

python eval.py \
    --dataset_name=r2g \
    --dataset_root=${DATA}/ \
    --eval_set=${SPLIT} \
    --checkpoint=${CKPT} \
    --output_dir=${SAVE_DIR} \
    --semantic_classes=12 \
    --input_channels 3 \
    --poly2seq \
    --seq_len 512 \
    --num_bins 32 \
    --disable_poly_refine \
    --dec_attn_concat_src \
    --ema4eval \
    --use_anchor \
    --per_token_sem_loss \
    --save_pred \
    --disable_sem_rich \
    --drop_wd