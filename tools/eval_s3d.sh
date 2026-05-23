#!/usr/bin/env bash

DATA=data/coco_s3d_bw
FOLDER=test

CKPT=hf:s3d-bw
python eval.py --dataset_name=stru3d \
   --dataset_root=${DATA} \
   --eval_set=${FOLDER} \
   --checkpoint=${CKPT} \
   --output_dir=eval_outputs/s3dbw_sem_results \
   --semantic_classes=19 \
   --input_channels 3 \
   --poly2seq \
   --seq_len 512 \
   --num_bins 32 \
   --disable_poly_refine \
   --dec_attn_concat_src \
   --use_anchor \
   --per_token_sem_loss \
   --ema4eval \
   --save_pred