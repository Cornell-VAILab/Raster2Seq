#!/usr/bin/env bash

DATA=data/R2G_hr_dataset_processed_v1
FOLDER=test 
CKPT=hf:raster2graph

python predict.py \
   --dataset_name=r2g \
   --dataset_root=${DATA}/${FOLDER} \
   --checkpoint=${CKPT} \
   --output_dir=pred_outputs/r2g_${FOLDER}_preds \
   --semantic_classes=13 \
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