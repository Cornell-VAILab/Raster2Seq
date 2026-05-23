#!/usr/bin/env bash

DATA=data/coco_cubicasa5k_nowalls_v4-1_refined/
FOLDER=test 
CKPT=hf:cubicasa5k

python predict.py \
    --dataset_name=cubicasa \
    --dataset_root=${DATA}/${FOLDER} \
    --checkpoint=${CKPT} \
    --output_dir=pred_outputs/cc5k_${FOLDER}_preds \
    --semantic_classes=12 \
    --input_channels 3 \
    --poly2seq \
    --seq_len 512 \
    --num_bins 32 \
    --disable_poly_refine \
    --dec_attn_concat_src \
    --per_token_sem_loss \
    --use_anchor \
    --ema4eval \
    --save_pred