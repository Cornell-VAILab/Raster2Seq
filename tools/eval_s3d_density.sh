#!/usr/bin/env bash

CKPT=hf:s3d-density
python eval.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --eval_set=test \
               --checkpoint=${CKPT} \
               --output_dir=eval_outputs/s3dd_sem_results \
               --semantic_classes=19 \
               --input_channels 1 \
               --poly2seq \
               --seq_len 512 \
               --num_bins 32 \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --ema4eval \
               --use_anchor \
               --per_token_sem_loss \
               --save_pred