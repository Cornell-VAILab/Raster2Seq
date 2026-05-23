#!/usr/bin/env bash

DATA=data/waffle_benchmark_processed/
SPLIT=test 

##### Trained on CubiCasa5K dataset, test on WAFFLE
NAME=cc5k_waffle_${SPLIT}_preds
SAVE_DIR=cross_eval_outputs/${NAME}
CKPT=hf:cubicasa5k
python predict.py \
   --dataset_root=${DATA}/${SPLIT} \
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
   --drop_wd \
   --one_color

python eval_seg.py evaluations/clipseg_eval/config.yaml 0 \
   ${SAVE_DIR}/checkpoints/jsons

##### Trained on Raster2Graph dataset, test on WAFFLE
NAME=r2g_waffle_${SPLIT}_preds
SAVE_DIR=cross_eval_outputs/${NAME}
CKPT=hf:raster2graph
python predict.py \
   --dataset_root=${DATA}/${SPLIT} \
   --checkpoint=${CKPT} \
   --output_dir=${SAVE_DIR} \
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
   --save_pred \
   --drop_wd \
   --one_color

python eval_seg.py evaluations/clipseg_eval/config.yaml 0 \
   ${SAVE_DIR}/checkpoints/jsons