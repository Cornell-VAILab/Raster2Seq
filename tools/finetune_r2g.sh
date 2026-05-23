#!/bin/bash

export NCCL_P2P_LEVEL=NVL

MASTER_PORT=25157
NUM_GPUS=2

CLS_COEFF=5
COO_COEFF=20
SEM_COEFF=1
SEQ_LEN=512
NUM_BINS=32
CONVERTER=v3

DATA=data/R2G_hr_dataset_processed_v1
JOB=r2g_sem_res256
PRETRAIN=checkpoints/r2g_res256_ep0849.pth
OUTPUT_DIR=save_models

WANDB_MODE=online torchrun --nproc_per_node=${NUM_GPUS} --master_port=$MASTER_PORT main_ddp.py --dataset_name=r2g \
               --dataset_root=${DATA} \
               --semantic_classes=13 \
               --job_name=${JOB} \
               --batch_size 56 \
               --input_channels=3 \
               --output_dir ${OUTPUT_DIR} \
               --poly2seq \
               --seq_len $SEQ_LEN \
               --num_bins ${NUM_BINS} \
               --ckpt_every_epoch=50 \
               --eval_every_epoch=50 \
               --label_smoothing 0.1 \
               --epochs 550 \
               --lr_drop '' \
               --cls_loss_coef ${CLS_COEFF} \
               --coords_loss_coef ${COO_COEFF} \
               --room_cls_loss_coef ${SEM_COEFF} \
               --resume ${OUTPUT_DIR}/${JOB}/checkpoint.pth \
               --ema4eval \
               --disable_poly_refine \
               --dec_attn_concat_src \
               --per_token_sem_loss \
               --jointly_train \
               --converter_version ${CONVERTER} \
               --use_anchor \
               --start_from_checkpoint ${PRETRAIN}