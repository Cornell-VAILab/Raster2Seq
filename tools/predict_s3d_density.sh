#!/usr/bin/env bash

DATA=data/stru3d/
FOLDER=test
CKPT=hf:s3d-density

python predict.py \
               --dataset_name=stru3d \
               --dataset_root=${DATA}/${FOLDER} \
               --checkpoint=${CKPT} \
               --output_dir=pred_outputs/s3dd_${FOLDER}_preds \
               --semantic_classes=19 \
               --input_channels 1 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --use_anchor \
               --ema4eval \
               --save_pred