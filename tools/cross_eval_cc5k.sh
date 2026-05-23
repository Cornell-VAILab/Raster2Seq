DATA=data/coco_cubicasa5k_nowalls_v4-1_refined
SPLIT=test 

NAME=r2g_cc5k_${SPLIT}_preds
SAVE_DIR=cross_eval_outputs/${NAME}
CKPT=hf:raster2graph

python eval.py \
   --dataset_name=cubicasa \
   --dataset_root=${DATA}/ \
   --eval_set=${SPLIT} \
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
   --disable_sem_rich \
   --drop_wd