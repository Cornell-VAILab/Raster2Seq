#!/usr/bin/env bash

CKPT=hf:cubicasa5k
python eval.py --dataset_name=cubicasa \
   --dataset_root=data/coco_cubicasa5k_nowalls_v4-1_refined/ \
   --eval_set=test \
   --checkpoint=${CKPT} \
   --output_dir=eval_outputs/cc5k_sem_results/ \
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
   --save_pred