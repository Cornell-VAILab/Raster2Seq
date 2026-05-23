import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import trange

from datasets import build_dataset
from engine import evaluate_floor, generate
from models import build_model
from raster2seq_hub import resolve_checkpoint_path


def get_args_parser():
    parser = argparse.ArgumentParser("Raster2Seq evaluation script", add_help=False)
    parser.add_argument("--batch_size", default=10, type=int)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input_channels", default=1, type=int)
    parser.add_argument("--image_norm", action="store_true")
    parser.add_argument("--eval_every_epoch", type=int, default=20)
    parser.add_argument("--ckpt_every_epoch", type=int, default=20)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--ignore_index", type=int, default=-1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--ema4eval", action="store_true")
    parser.add_argument("--measure_time", action="store_true")
    parser.add_argument("--disable_sampling_cache", action="store_true")
    parser.add_argument("--use_anchor", action="store_true")
    parser.add_argument("--drop_wd", action="store_true")
    parser.add_argument("--iou_thres", type=float, default=0.5)
    parser.add_argument("--disable_sem_rich", action="store_true")
    parser.add_argument("--wd_only", action="store_true")
    parser.add_argument("--disable_image_transform", action="store_true")
    parser.add_argument("--num_subset_images", type=int, default=-1)
    parser.add_argument("--converter_version", type=str, default="v1")
    parser.add_argument("--inject_cls_embed", action="store_true")

    # raster2seq
    parser.add_argument("--poly2seq", action="store_true")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_bins", type=int, default=64)
    parser.add_argument("--pre_decoder_pos_embed", action="store_true")
    parser.add_argument("--learnable_dec_pe", action="store_true")
    parser.add_argument("--dec_qkv_proj", action="store_true")
    parser.add_argument("--dec_attn_concat_src", action="store_true")
    parser.add_argument("--per_token_sem_loss", action="store_true")
    parser.add_argument("--add_cls_token", action="store_true")

    # backbone
    parser.add_argument("--backbone", default="resnet50", type=str, help="Name of the convolutional backbone to use")
    parser.add_argument("--lr_backbone", default=0, type=float)
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument("--position_embedding_scale", default=2 * np.pi, type=float, help="position / size * scale")
    parser.add_argument("--num_feature_levels", default=4, type=int, help="number of feature levels")

    # Transformer
    parser.add_argument("--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument(
        "--dim_feedforward",
        default=1024,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim", default=256, type=int, help="Size of the embeddings (dimension of the transformer)"
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions"
    )
    parser.add_argument(
        "--num_queries",
        default=800,
        type=int,
        help="Number of query slots (num_polys * max. number of corner per poly)",
    )
    parser.add_argument("--num_polys", default=20, type=int, help="Number of maximum number of room polygons")
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)

    parser.add_argument(
        "--query_pos_type",
        default="sine",
        type=str,
        choices=("static", "sine", "none"),
        help="Type of query pos in decoder - \
                        1. static: same setting with DETR and Deformable-DETR, the query_pos is the same for all layers \
                        2. sine: since embedding from reference points (so if references points update, query_pos also \
                        3. none: remove query_pos",
    )
    parser.add_argument(
        "--with_poly_refine",
        default=True,
        action="store_true",
        help="iteratively refine reference points (i.e. positional part of polygon queries)",
    )
    parser.add_argument(
        "--masked_attn",
        default=False,
        action="store_true",
        help="if true, the query in one room will not be allowed to attend other room",
    )
    parser.add_argument(
        "--semantic_classes",
        default=-1,
        type=int,
        help="Number of classes for semantically-rich floorplan:  \
                        1. default -1 means non-semantic floorplan \
                        2. 19 for Structured3D: 16 room types + 1 door + 1 window + 1 empty",
    )
    parser.add_argument(
        "--disable_poly_refine",
        action="store_true",
        help="iteratively refine reference points (i.e. positional part of polygon queries)",
    )

    # aux
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_true",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # dataset parameters
    parser.add_argument("--dataset_name", default="stru3d")
    parser.add_argument("--dataset_root", default="data/stru3d", type=str)
    parser.add_argument("--eval_set", default="test", type=str)

    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint", default="checkpoints/roomformer_scenecad.pth", help="resume from checkpoint")
    parser.add_argument("--output_dir", default="eval_stru3d", help="path where to save result")

    # visualization options
    parser.add_argument("--plot_pred", default=True, type=bool, help="plot predicted floorplan")
    parser.add_argument(
        "--plot_density", default=True, type=bool, help="plot predicited room polygons overlaid on the density map"
    )
    parser.add_argument("--plot_gt", default=True, type=bool, help="plot ground truth floorplan")
    parser.add_argument("--save_pred", action="store_true", help="save_pred")

    return parser


def main(args):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build dataset and dataloader
    dataset_eval = build_dataset(image_set=args.eval_set, args=args)

    tokenizer = None
    if args.poly2seq:
        args.vocab_size = dataset_eval.get_vocab_size()
        tokenizer = dataset_eval.get_tokenizer()

    # overfit one sample
    if args.debug:
        dataset_eval = torch.utils.data.Subset(dataset_eval, [2])
        dataset_eval[0]

    if args.num_subset_images > 0 and args.num_subset_images < len(dataset_eval):
        dataset_eval = torch.utils.data.Subset(dataset_eval, range(args.num_subset_images))

    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch, None

    data_loader_eval = DataLoader(
        dataset_eval,
        args.batch_size,
        sampler=sampler_eval,
        drop_last=False,
        collate_fn=trivial_batch_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # build model
    model = build_model(args, train=False, tokenizer=tokenizer)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    for n, p in model.named_parameters():
        print(n)

    output_dir = Path(args.output_dir)

    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.ema4eval:
        ckpt_state_dict = copy.deepcopy(checkpoint["ema"])
    else:
        ckpt_state_dict = copy.deepcopy(checkpoint["model"])
    for key, value in checkpoint["model"].items():
        if key.startswith("module."):
            ckpt_state_dict[key[7:]] = checkpoint["model"][key]
            del ckpt_state_dict[key]
    missing_keys, unexpected_keys = model.load_state_dict(ckpt_state_dict, strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith("total_params") or k.endswith("total_ops"))]
    if len(missing_keys) > 0:
        print("Missing Keys: {}".format(missing_keys))
    if len(unexpected_keys) > 0:
        print("Unexpected Keys: {}".format(unexpected_keys))

    # disable grad
    for param in model.parameters():
        param.requires_grad = False

    if args.measure_time:
        # images = torch.rand(args.batch_size, 3, args.image_size, args.image_size).to(device)
        images = (
            torch.from_numpy(np.array(Image.open("data/coco_s3d_bw/val/03006.png").convert("RGB")))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            / 255.0
        )
        # INIT LOGGERS
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 50
        timings = np.zeros((repetitions, 1))
        if args.poly2seq:
            model = torch.compile(model)  # compile model is not compatible with RoomFormer
        # GPU-WARM-UP
        for _ in trange(10, desc="GPU-WARM-UP"):
            if not args.poly2seq:
                _ = model(images)
            else:
                _ = model.forward_inference(images)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in trange(repetitions):
                starter.record()
                _ = generate(
                    model,
                    images,
                    semantic_rich=args.semantic_classes > 0,
                    use_cache=True,
                    per_token_sem_loss=args.per_token_sem_loss,
                    drop_wd=args.drop_wd,
                    poly2seq=args.poly2seq,
                )
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print("Inference time: {:.2f}+/-{:.2f}ms".format(mean_syn, std_syn))
        exit(0)

    # save_dir = os.path.join(os.path.dirname(args.checkpoint), output_dir)
    # save_dir = os.path.join(output_dir, os.path.dirname(args.checkpoint).split('/')[-1])
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    evaluate_floor(
        model,
        args.dataset_name,
        data_loader_eval,
        device,
        save_dir,
        plot_pred=args.plot_pred,
        plot_density=args.plot_density,
        plot_gt=args.plot_gt,
        semantic_rich=(args.semantic_classes > 0 and not args.disable_sem_rich),
        save_pred=args.save_pred,
        per_token_sem_loss=args.per_token_sem_loss,
        iou_thres=args.iou_thres,
        poly2seq=args.poly2seq,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Raster2Seq evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()

    if args.debug:
        args.batch_size = 1
    if args.disable_poly_refine:
        args.with_poly_refine = False

    main(args)
