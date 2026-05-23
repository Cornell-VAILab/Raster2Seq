#!/usr/bin/env bash

DATA=data/coco_s3d_bw/
FOLDER=test
CKPT=hf:s3d-bw

python predict.py \
    --dataset_name=stru3d \
    --dataset_root=${DATA}/${FOLDER} \
    --checkpoint=${CKPT} \
    --output_dir=pred_outputs/s3d_${FOLDER}_preds \
    --semantic_classes=19 \
    --input_channels 3 \
    --poly2seq \
    --seq_len 512 \
    --num_bins 32 \
    --disable_poly_refine \
    --dec_attn_concat_src \
    --per_token_sem_loss \
    --use_anchor \
    --ema4eval \
    --save_pred \
    --debug