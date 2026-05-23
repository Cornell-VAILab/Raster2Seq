# !/usr/bin/env bash

CKPT=hf:raster2graph
python eval.py --dataset_name=r2g \
   --dataset_root=data/R2G_hr_dataset_processed_v1 \
   --eval_set=test \
   --checkpoint=${CKPT} \
   --output_dir=eval_outputs/r2g_sem_results \
   --semantic_classes=13 \
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