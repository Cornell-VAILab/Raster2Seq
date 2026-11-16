import copy
import json
import math
import os
import sys
from typing import Iterable

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon

import util.misc as utils

# Add evaluations folder relative to this file's location
sys.path.append(os.path.join(os.path.dirname(__file__), "evaluations"))
from rplan_eval.Evaluator import Evaluator_RPlan
from s3d_floorplan_eval.DataRW.S3DRW import S3DRW
from s3d_floorplan_eval.DataRW.wrong_annotatios import wrong_s3d_annotations_list
from s3d_floorplan_eval.Evaluator.Evaluator import Evaluator
from s3d_floorplan_eval.options import MCSSOptions

from datasets import get_dataset_class_labels
from datasets.poly_data import TokenType
from util.eval_utils import compute_f1
from util.plot_utils import (
    concat_floorplan_maps,
    plot_density_map,
    plot_floorplan_with_regions,
    plot_semantic_rich_floorplan_opencv,
    sort_polygons_by_matching,
)
from util.poly_ops import pad_gt_polys

# add specific helper for lines and per-label scoring
from evaluations.line_eval import line_scores, label_scores

options = MCSSOptions()
opts = options.parse()


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    poly2seq: bool = False,
    ema_model=None,
    **kwargs,
):
    """
    Trains the model for one epoch using the provided data loader, criterion, and optimizer.
    This function iterates over the data loader, computes losses, performs backpropagation,
    applies gradient clipping if specified, updates the model parameters, and optionally
    updates an EMA model. It logs various metrics including loss, learning rate, and gradient norm.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        criterion (torch.nn.Module): The loss criterion used to compute the loss.
        data_loader (Iterable): An iterable data loader yielding batches of inputs and extras.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.
        device (torch.device): The device (CPU or GPU) on which to perform computations.
        epoch (int): The current epoch number, used for logging.
        max_norm (float, optional): Maximum norm for gradient clipping. If 0, no clipping is applied. Defaults to 0.
        poly2seq (bool, optional): If True, uses batched_extras as room_targets and passes them to the model. Defaults to False.
        ema_model (optional): Exponential moving average model to update, if provided. Defaults to None.
        **kwargs: Additional keyword arguments, such as 'drop_rate' for padding ground truth polygons.
    Returns:
        dict: A dictionary containing the global averages of logged metrics (e.g., loss, lr, grad_norm).
    """

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad_norm", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    model_obj = model if not hasattr(model, "module") else model.module

    for batched_inputs, batched_extras in metric_logger.log_every(data_loader, print_freq, header):
        samples = [x["image"].to(device) for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        if not poly2seq:
            room_targets = pad_gt_polys(
                gt_instances,
                model_obj.num_queries_per_poly,
                samples[0].shape[1],
                drop_rate=kwargs.get("drop_rate", 0.0),
                device=device,
            )
            outputs = model(samples)
        else:
            for key in batched_extras.keys():
                batched_extras[key] = batched_extras[key].to(device)
            room_targets = batched_extras
            outputs = model(samples, batched_extras)

        loss_dict = criterion(outputs, room_targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_unscaled = {f"{k}_unscaled": v for k, v in loss_dict.items()}
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        if ema_model is not None:
            utils.update_ema(ema_model, model.module, 0.999)

        metric_logger.update(loss=loss_value, **loss_dict_scaled, **loss_dict_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    dataset_name,
    data_loader,
    device,
    plot_density=False,
    output_dir=None,
    epoch=None,
    poly2seq: bool = False,
    add_cls_token=False,
    per_token_sem_loss=False,
    wd_as_line=True,
):
    """
    Evaluates the model on a given dataset during training, computing losses and various metrics such as room IoU,
    precision, recall, corner precision/recall, angle precision/recall, and semantic metrics if applicable.

    This function supports two evaluation modes:
    - Non-poly2seq mode (poly2seq=False): RoomFormer evaluation with fg_mask filtering
    - Poly2seq mode (poly2seq=True): Raster2Seq evaluation with sequence-based predictions

    Args:
        model: The neural network model to evaluate.
        criterion: The loss criterion used for evaluation.
        dataset_name (str): Name of the dataset (e.g., "stru3d", "cubicasa", "r2g", "waffle", "osm).
        data_loader: DataLoader providing batches of input data.
        device: The device (e.g., CPU or GPU) to run the evaluation on.
        plot_density (bool, optional): If True, plots a density map for the last sample. Defaults to False.
        output_dir (str, optional): Directory to save output plots if plot_density is True. Defaults to None.
        epoch (int, optional): Current epoch number, used in plot filenames. Defaults to None.
        poly2seq (bool, optional): If True, uses sequence-based prediction mode with forward_inference.
                                   If False, uses mask-based filtering with fg_mask. Defaults to False.
        add_cls_token (bool, optional): If True, accounts for class tokens in sequence processing.
                                        Only used when poly2seq=True. Defaults to False.
        per_token_sem_loss (bool, optional): If True, computes semantic loss per token using voting.
                                             Only used when poly2seq=True. Defaults to False.
        wd_as_line (bool, optional): If True, treats windows/doors as lines based on corner count.
                                     If False, uses semantic class. Only used when poly2seq=True. Defaults to True.

    Returns:
        dict: A dictionary containing averaged evaluation statistics, including losses and metrics like room_iou,
        room_prec, room_rec, corner_prec, corner_rec, angles_prec, angles_rec, and semantic metrics if applicable.
    """

    model.eval()
    criterion.eval()

    if dataset_name == "stru3d":
        door_window_index = [16, 17]
    elif dataset_name == "cubicasa":
        door_window_index = [10, 9]
    else:
        door_window_index = []

    if dataset_name == "osm":
        wd_as_line = False

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    model_obj = model if not hasattr(model, "module") else model.module

    # determine lines differently based on stage 1 or 2 output:
    # - Stage 1 with --geom_classes: the model predicts 0 = polygon, 1 = line, so "line" is class 1.
    # Stage 2: the model predicts 33 labels. Lines are several of those (street, path, highway)
    line_classes = set()
    if dataset_name == "osm":
        ds = data_loader.dataset
        line_classes = {1} if getattr(ds, "geom_classes", False) else {
            a["category_id"] for a in ds.coco.dataset["annotations"] if a.get("geom_type") == "line"}

    for batched_inputs, batched_extras in metric_logger.log_every(data_loader, 10, header):
        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]

        if not poly2seq:
            room_targets = pad_gt_polys(
                gt_instances, model_obj.num_queries_per_poly, samples[0].shape[1], drop_rate=0.0, device=device
            )
            outputs = model(samples)
        else:
            for key in batched_extras.keys():
                batched_extras[key] = batched_extras[key].to(device)
            room_targets = batched_extras
            outputs = model(samples, batched_extras)

        image_size = samples[0].size(2)
        loss_dict = criterion(outputs, room_targets)
        weight_dict = criterion.weight_dict

        if poly2seq:
            outputs = model_obj.forward_inference(samples)
            pred_corners = outputs["gen_out"]
            fg_mask = None
        else:
            pred_logits = outputs["pred_logits"]
            pred_corners = outputs["pred_coords"]
            # TODO: double check
            fg_mask = torch.sigmoid(pred_logits) > 0.5  # select valid corners

        bs = outputs["pred_logits"].shape[0]

        if "pred_room_logits" in outputs:
            pred_room_logits = outputs["pred_room_logits"]
            prob = torch.nn.functional.softmax(pred_room_logits, -1)
            _, pred_room_label = prob[..., :-1].max(-1)

        # process per scene
        for i in range(bs):
            # Prepare ground truth data
            gt_polys, gt_polys_types = [], []
            gt_window_doors = []
            gt_window_doors_types = []
            for gt_poly, gt_id in zip(
                gt_instances[i].gt_masks.polygons, gt_instances[i].gt_classes.detach().cpu().tolist()
            ):
                gt_poly = gt_poly[0].reshape(-1, 2).astype(np.int32)
                if gt_id in door_window_index:
                    gt_window_doors.append(gt_poly)
                    gt_window_doors_types.append(gt_id)
                else:
                    gt_polys.append(gt_poly)
                    gt_polys_types.append(gt_id)

            # Create evaluator based on dataset
            if dataset_name == "stru3d":
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode="online_eval")
                evaluator = Evaluator(curr_data_rw, curr_opts, disable_overlapping_filter=poly2seq)
            elif dataset_name in ["cubicasa", "r2g", "waffle", "osm"]:
                evaluator = Evaluator_RPlan(disable_overlapping_filter=poly2seq, wd_as_line=wd_as_line)

            print("Running Evaluation for scene %s" % scene_ids[i])

            room_polys = []
            semantic_rich = "pred_room_logits" in outputs

            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []

            scene_outputs = _process_predictions(
                pred_corners,
                i,
                semantic_rich,
                poly2seq,
                fg_mask,
                image_size,
                pred_room_label if semantic_rich else None,
                pred_room_logits if semantic_rich else None,
                add_cls_token,
                per_token_sem_loss,
                wd_as_line,
                door_window_index,
                dataset_name,
            )
            room_polys = scene_outputs["room_polys"]
            room_types = scene_outputs["room_types"]
            window_doors = scene_outputs["window_doors"]
            window_doors_types = scene_outputs["window_doors_types"]
            pred_room_label_per_scene = scene_outputs["pred_room_label_per_scene"]

            if dataset_name == "osm" and semantic_rich:
                # lines: buffered into thin polygons and scored on their own; the evaluator below gets polygons only
                metric_logger.update(**line_scores(room_polys, room_types, gt_polys, gt_polys_types, line_classes,
                                                   (image_size, image_size)))
                keep = [t not in line_classes for t in room_types]
                room_polys = [p for p, k in zip(room_polys, keep) if k]
                room_types = [t for t, k in zip(room_types, keep) if k]
                gt_keep = [t not in line_classes for t in gt_polys_types]
                gt_polys = [p for p, k in zip(gt_polys, gt_keep) if k]
                gt_polys_types = [t for t, k in zip(gt_polys_types, gt_keep) if k]

            if dataset_name == "stru3d":
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                        room_polys=room_polys,
                        room_types=room_types,
                        window_door_lines=window_doors,
                        window_door_lines_types=window_doors_types,
                    )
            elif dataset_name in ["cubicasa", "r2g", "waffle", "osm"]:
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                        room_polys=room_polys,
                        gt_polys=gt_polys,
                        room_types=None,
                        gt_polys_types=gt_polys_types,
                        img_size=(image_size, image_size),
                    )
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                        room_polys=room_polys,
                        gt_polys=gt_polys,
                        room_types=room_types,
                        gt_polys_types=gt_polys_types,
                        window_door_lines=window_doors,
                        gt_window_doors_list=gt_window_doors,
                        window_door_lines_types=window_doors_types,
                        gt_window_doors_type_list=gt_window_doors_types,
                        img_size=(image_size, image_size),
                    )

            if "room_iou" in quant_result_dict_scene:
                metric_logger.update(room_iou=quant_result_dict_scene["room_iou"])

            metric_logger.update(room_prec=quant_result_dict_scene["room_prec"])
            metric_logger.update(room_rec=quant_result_dict_scene["room_rec"])
            metric_logger.update(corner_prec=quant_result_dict_scene["corner_prec"])
            metric_logger.update(corner_rec=quant_result_dict_scene["corner_rec"])
            metric_logger.update(angles_prec=quant_result_dict_scene["angles_prec"])
            metric_logger.update(angles_rec=quant_result_dict_scene["angles_rec"])

            if semantic_rich:
                metric_logger.update(room_sem_prec=quant_result_dict_scene["room_sem_prec"])
                metric_logger.update(room_sem_rec=quant_result_dict_scene["room_sem_rec"])
                metric_logger.update(window_door_prec=quant_result_dict_scene["window_door_prec"])
                metric_logger.update(window_door_rec=quant_result_dict_scene["window_door_rec"])

        # plot last sample
        if plot_density and len(room_polys) > 0:
            pred_room_map = plot_density_map(samples[i], image_size, room_polys, pred_room_label_per_scene)
            cv2.imwrite(os.path.join(output_dir, "{}_pred_room_map_{}.png".format(scene_ids[i], epoch)), pred_room_map)
            plot_density = False  # only plot once

        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f"{k}_unscaled": v for k, v in loss_dict.items()}
        metric_logger.update(loss=sum(loss_dict_scaled.values()), **loss_dict_scaled, **loss_dict_unscaled)

    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


def _process_predictions(
    pred_corners,
    i,
    semantic_rich,
    poly2seq,
    fg_mask,
    image_size,
    pred_room_label,
    pred_room_logits,
    add_cls_token,
    per_token_sem_loss,
    wd_as_line,
    door_window_index,
    dataset_name,
):
    """
    Processes predictions for room layouts, extracting polygons for rooms, windows, and doors.
    This function handles two main modes: non-poly2seq (mask-based) and poly2seq (sequence-based).
    It filters and validates predicted corners based on various conditions, computes room types
    and window/door types if semantic information is rich, and ensures polygons meet area thresholds.

    Args:
        pred_corners (list or tensor): Predicted corner coordinates for scenes or sequences.
        i (int): Index of the current scene in the batch.
        semantic_rich (bool): Whether to include semantic information (room types, window/door types).
        poly2seq (bool): Whether the predictions are in sequence format (poly2seq mode).
        fg_mask (tensor): Foreground masks for valid corners per room (used in non-poly2seq mode).
        image_size (int): Size of the image (used for scaling coordinates).
        pred_room_label (tensor): Predicted room labels for the scene.
        pred_room_logits (tensor): Logits for room predictions.
        add_cls_token (int): Number of CLS tokens added in sequences (used in poly2seq mode).
        per_token_sem_loss (bool): Whether to compute semantic loss per token (affects class aggregation).
        wd_as_line (bool): Whether to treat windows/doors as lines (affects classification logic).
        door_window_index (list or dict): Indices or mapping for door/window classes.
        dataset_name (str): Name of the dataset being processed.

    Returns:
        dict: A dictionary containing:
            - "room_polys" (list of np.ndarray): List of valid room polygons (scaled and rounded).
            - "room_types" (list or None): List of room types if semantic_rich, else None.
            - "window_doors" (list or None): List of window/door polygons if semantic_rich, else None.
            - "window_doors_types" (list or None): List of window/door types if semantic_rich, else None.
            - "pred_room_label_per_scene" (list): Processed room labels for the scene.
    """

    np_softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    pred_corners_per_scene = pred_corners[i]
    room_polys = []

    if semantic_rich:
        room_types = []
        window_doors = []
        window_doors_types = []

        pred_room_label_per_scene = pred_room_label[i].cpu().numpy()
        pred_room_logit_per_scene = pred_room_logits[i].cpu().numpy()

    # Process predictions based on mode
    if not poly2seq:
        fg_mask_per_scene = fg_mask[i]

        # process per room
        for j in range(fg_mask_per_scene.shape[0]):
            fg_mask_per_room = fg_mask_per_scene[j]
            pred_corners_per_room = pred_corners_per_scene[j]
            valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]
            if len(valid_corners_per_room) > 0:
                corners = (valid_corners_per_room * (image_size - 1)).cpu().numpy()
                corners = np.around(corners).astype(np.int32)

                if not semantic_rich:
                    # only regular rooms
                    if len(corners) >= 4 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                else:
                    # regular rooms
                    if len(corners) >= 3 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                        room_types.append(pred_room_label_per_scene[j])
                    # window / door
                    elif len(corners) == 2:
                        window_doors.append(corners)
                        window_doors_types.append(pred_room_label_per_scene[j])

        if not semantic_rich:
            pred_room_label_per_scene = len(room_polys) * [-1]
    else:
        # poly2seq mode: sequence-based processing
        all_room_polys = []
        tmp = []
        all_length_list = [0]

        for j in range(len(pred_corners_per_scene)):
            token = pred_corners_per_scene[j]
            if isinstance(token, int):
                # Handle separator, end-of-sequence, or class tokens
                # New format: sep=1, eos=2, cls=3 (following TokenType)
                # Old format: sep=2, eos=-1, cls=-1
                if (
                    token == TokenType.sep.value
                    or token == TokenType.eos.value
                    or token == TokenType.cls.value
                    or token == -1
                ):
                    if tmp:
                        all_room_polys.append(tmp)
                        all_length_list.append(len(tmp) + 1 + add_cls_token)
                        tmp = []
                
                if token == -1 or token == TokenType.eos.value:
                    break
                continue
            tmp.append(token)

        if len(tmp):
            all_room_polys.append(tmp)
            all_length_list.append(len(tmp) + 1 + add_cls_token)

        start_poly_indices = np.cumsum(all_length_list)

        final_pred_classes = []
        for j, poly in enumerate(all_room_polys):
            if len(poly) < 2:
                continue
            corners = np.array(poly, dtype=np.float32) * (image_size - 1)
            corners = np.around(corners).astype(np.int32)

            if not semantic_rich:
                # only regular rooms
                if len(corners) >= 4 and Polygon(corners).area >= 100:
                    room_polys.append(corners)
            else:
                if per_token_sem_loss:
                    pred_classes, counts = np.unique(
                        pred_room_label_per_scene[start_poly_indices[j] : start_poly_indices[j + 1]][:-1],
                        return_counts=True,
                    )
                    pred_class = pred_classes[np.argmax(counts)]
                    pred_logit = pred_room_logit_per_scene[start_poly_indices[j] : start_poly_indices[j + 1]][:-1]
                else:
                    pred_class = pred_room_label_per_scene[
                        start_poly_indices[j + 1] - 1
                    ]  # get last cls token in the seq
                final_pred_classes.append(pred_class)

                if wd_as_line:
                    # regular rooms
                    if len(corners) >= 3 and Polygon(corners).area >= 100:
                        room_polys.append(corners)
                        room_types.append(pred_class)
                    # window / door
                    elif len(corners) == 2:
                        window_doors.append(corners)
                        if (
                            door_window_index is not None
                            and pred_class not in door_window_index
                            and dataset_name != "r2g"
                        ):
                            wd_prob = np_softmax(pred_logit[:, door_window_index].sum(0))
                            pred_class = door_window_index[wd_prob.argmax()]
                        window_doors_types.append(pred_class)
                else:
                    # regular rooms
                    if door_window_index is not None and pred_class not in door_window_index:
                        room_polys.append(corners)
                        room_types.append(pred_class)
                    else:
                        window_doors.append(corners)
                        window_doors_types.append(pred_class)

        if not semantic_rich:
            pred_room_label_per_scene = len(all_room_polys) * [-1]
        else:
            pred_room_label_per_scene = final_pred_classes

    return {
        "room_polys": room_polys,
        "room_types": room_types if semantic_rich else None,
        "window_doors": window_doors if semantic_rich else None,
        "window_doors_types": window_doors_types if semantic_rich else None,
        "pred_room_label_per_scene": pred_room_label_per_scene,
    }


@torch.no_grad()
def evaluate_floor(
    model,
    dataset_name,
    data_loader,
    device,
    output_dir,
    plot_pred=True,
    plot_density=True,
    plot_gt=True,
    semantic_rich=False,
    save_pred=False,
    poly2seq: bool = False,
    add_cls_token=False,
    per_token_sem_loss=False,
    iou_thres=0.5,
    wd_as_line=True,
    per_label=False, # new to handle per-label metrics
):
    """
    Evaluate the model on a given dataset at testing.

    This function supports two evaluation modes:
    - Non-poly2seq mode (poly2seq=False): RoomFormer evaluation with fg_mask filtering
    - Poly2seq mode (poly2seq=True): Raster2Seq evaluation with sequence-based predictions

    This function processes the dataset in batches, performs inference using the provided model,
    and computes quantitative metrics such as precision, recall, and F1-score for rooms, corners,
    and angles. It supports semantic-rich evaluation including room types and window/door detection
    for specific datasets. Optionally, it can plot predicted floorplans, density maps, and save
    predictions as JSON files.

    Args:
        model: The trained model to evaluate (e.g., a PyTorch model).
        dataset_name (str): Name of the dataset (e.g., "stru3d", "cubicasa", "r2g", "waffle", "osm").
        data_loader: PyTorch DataLoader providing batched inputs and extras.
        device: Device to run the model on (e.g., torch.device('cuda') or 'cpu').
        output_dir (str): Directory path to save output files (plots, JSONs, results).
        plot_pred (bool, optional): If True, plot and save predicted floorplans. Default is True.
        plot_density (bool, optional): If True, plot and save density maps. Default is True.
        plot_gt (bool, optional): If True, save ground truth images. Default is True.
        semantic_rich (bool, optional): If True, perform semantic evaluation including room types and window/door lines. Default is False.
        save_pred (bool, optional): If True, save predicted polygons and metrics as JSON files. Default is False.
        poly2seq (bool, optional): If True, uses sequence-based prediction mode with forward_inference.
                                   If False, uses mask-based filtering with fg_mask. Defaults to False.
        add_cls_token (bool, optional): If True, accounts for class tokens in sequence processing.
                                        Only used when poly2seq=True. Defaults to False.
        per_token_sem_loss (bool, optional): If True, computes semantic loss per token using voting.
                                             Only used when poly2seq=True. Defaults to False.
        iou_thres (float, optional): IoU threshold for matching predictions to ground truth. Default is 0.5.
        wd_as_line (bool, optional): If True, treats windows/doors as lines based on corner count.
                                     If False, uses semantic class. Only used when poly2seq=True. Defaults to True.

    Returns:
        None. The function prints the aggregated quantitative results to the console and saves them to 'results.txt' in output_dir.
    """
    model.eval()

    if dataset_name == "stru3d":
        door_window_index = [16, 17]
    elif dataset_name == "cubicasa":
        door_window_index = [10, 9]
    elif dataset_name == "waffle":
        door_window_index = []  # [1, 2]
    else:
        door_window_index = []

    if dataset_name == "osm":
        wd_as_line = False
        

    metric_category = ["room", "corner", "angles"]
    if semantic_rich:
        metric_category += ["room_sem", "window_door"]

    quant_result_dict = None
    scene_counter = 0
    merge = False  # Only used in poly2seq mode for merged plot output

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # handles lines differently based on stage 1 vs. stage 2
    ds = data_loader.dataset
    line_classes = {1} if getattr(ds, "geom_classes", False) else {
        a["category_id"] for a in ds.coco.dataset["annotations"] if a.get("geom_type") == "line"}
    line_results = []
    label_results = [] # --per_label_eval, initialize, will be used later

    for batched_inputs, batched_extras in data_loader:
        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]

        image_size = samples[0].size(2)

        # Get predictions based on mode
        if poly2seq:
            outputs = model.forward_inference(samples)
            pred_corners = outputs["gen_out"]
            fg_mask = None
        else:
            outputs = model(samples)
            pred_logits = outputs["pred_logits"]
            pred_corners = outputs["pred_coords"]
            fg_mask = torch.sigmoid(pred_logits) > 0.5  # select valid corners

        bs = outputs["pred_logits"].shape[0]

        if "pred_room_logits" in outputs:
            pred_room_logits = outputs["pred_room_logits"]
            prob = torch.nn.functional.softmax(pred_room_logits, -1)
            _, pred_room_label = prob[..., :-1].max(-1)

        # process per scene
        for i in range(bs):
            # Prepare ground truth data
            gt_polys, gt_polys_types = [], []
            gt_window_doors = []
            gt_window_doors_types = []
            for gt_poly, gt_id in zip(
                gt_instances[i].gt_masks.polygons, gt_instances[i].gt_classes.detach().cpu().tolist()
            ):
                gt_poly = gt_poly[0].reshape(-1, 2).astype(np.int32)
                if gt_id in door_window_index:
                    gt_window_doors.append(gt_poly)
                    gt_window_doors_types.append(gt_id)
                else:
                    gt_polys.append(gt_poly)
                    gt_polys_types.append(gt_id)

            # Create evaluator based on dataset
            if dataset_name == "stru3d":
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode="test")
                evaluator = Evaluator(curr_data_rw, curr_opts, disable_overlapping_filter=poly2seq)
            elif dataset_name in ["cubicasa", "waffle", "r2g", "osm"]:
                evaluator = Evaluator_RPlan(
                    disable_overlapping_filter=poly2seq, iou_thres=iou_thres, wd_as_line=wd_as_line
                )

            print("Running Evaluation for scene %s" % scene_ids[i])

            scene_outputs = _process_predictions(
                pred_corners,
                i,
                semantic_rich,
                poly2seq,
                fg_mask,
                image_size,
                pred_room_label if semantic_rich else None,
                pred_room_logits if semantic_rich else None,
                add_cls_token,
                per_token_sem_loss,
                wd_as_line,
                door_window_index,
                dataset_name,
            )
            room_polys = scene_outputs["room_polys"]
            room_types = scene_outputs["room_types"]
            window_doors = scene_outputs["window_doors"]
            window_doors_types = scene_outputs["window_doors_types"]
            pred_room_label_per_scene = scene_outputs["pred_room_label_per_scene"]

            pred_lines = gt_lines = [] # initialize to empty for other datasets
            if dataset_name == "osm" and semantic_rich:
                # lines: buffered into thin polygons and scored on their own; the evaluator below gets polygons only
                scores = line_scores(room_polys, room_types, gt_polys, gt_polys_types, line_classes,
                                     (image_size, image_size))
                if scores:
                    # adding line scores
                    line_results.append(scores)

                # adding per-label scores
                if per_label:  # per-label precision / recall, before lines are removed below
                    label_results.append(label_scores(room_polys, room_types, gt_polys, gt_polys_types, line_classes,
                                                      (image_size, image_size)))

                # keep line predictions and gt (for plots)
                pred_lines = [(p, t) for p, t in zip(room_polys, room_types) if t in line_classes]
                gt_lines = [(p, t) for p, t in zip(gt_polys, gt_polys_types) if t in line_classes]

                keep = [t not in line_classes for t in room_types]
                room_polys = [p for p, k in zip(room_polys, keep) if k]
                room_types = [t for t, k in zip(room_types, keep) if k]
                gt_keep = [t not in line_classes for t in gt_polys_types]
                gt_polys = [p for p, k in zip(gt_polys, gt_keep) if k]
                gt_polys_types = [t for t, k in zip(gt_polys_types, gt_keep) if k]

            if dataset_name == "stru3d":
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                        room_polys=room_polys,
                        room_types=room_types,
                        window_door_lines=window_doors,
                        window_door_lines_types=window_doors_types,
                    )
            elif dataset_name in ["cubicasa", "waffle", "r2g", "osm"]:
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                        room_polys=room_polys,
                        gt_polys=gt_polys,
                        room_types=None,
                        gt_polys_types=gt_polys_types,
                        img_size=(image_size, image_size),
                    )
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                        room_polys=room_polys,
                        gt_polys=gt_polys,
                        room_types=room_types,
                        gt_polys_types=gt_polys_types,
                        window_door_lines=window_doors,
                        gt_window_doors_list=gt_window_doors,
                        window_door_lines_types=window_doors_types,
                        gt_window_doors_type_list=gt_window_doors_types,
                        img_size=(image_size, image_size),
                    )

            if quant_result_dict is None:
                quant_result_dict = quant_result_dict_scene
            else:
                for k in quant_result_dict.keys():
                    quant_result_dict[k] += quant_result_dict_scene[k]

            scene_counter += 1

            # plot regular room floorplan
            gt_room_polys = [np.array(poly) for poly in gt_polys]
            room_polys = [np.array(poly) for poly in room_polys]

            if "gt_polys_sorted_indcs" in quant_result_dict_scene:
                gt_polys_sorted_indcs = quant_result_dict_scene["gt_polys_sorted_indcs"]
                del quant_result_dict_scene["gt_polys_sorted_indcs"]
                gt_room_polys = [gt_room_polys[ind] for ind in gt_polys_sorted_indcs]

            if "pred2gt_indices" in quant_result_dict_scene:
                pred2gt_indices = quant_result_dict_scene["pred2gt_indices"]
                del quant_result_dict_scene["pred2gt_indices"]
                room_polys, gt_room_polys, pred_mask, gt_mask = sort_polygons_by_matching(
                    pred2gt_indices, room_polys, gt_room_polys
                )
            else:
                pred_mask, gt_mask = None, None

            prec, rec = quant_result_dict_scene["room_prec"], quant_result_dict_scene["room_rec"]
            f1 = 2 * prec * rec / (prec + rec + 1e-5)
            missing_rate = quant_result_dict_scene["room_missing_ratio"]
            plot_statistics = {
                "f1": f1,
                "prec": prec,
                "rec": rec,
                "missing_rate": missing_rate,
                "num_preds": len(room_polys),
                "num_gt": len(gt_polys),
                "num_matched_preds": sum([x != -1 for x in pred2gt_indices]),
            }

            if plot_pred:
                # account for lines separately
                pl = [np.array(p) for p, _ in pred_lines]
                gl = [np.array(p) for p, _ in gt_lines]
                gt_floorplan_map = plot_floorplan_with_regions(
                    gt_room_polys + gl, matching_labels=None if gt_mask is None else gt_mask + [True] * len(gl),
                    base_scale=image_size, scale=1024, is_line=[False] * len(gt_room_polys) + [True] * len(gl))
                floorplan_map = plot_floorplan_with_regions(
                    room_polys + pl, matching_labels=None if pred_mask is None else pred_mask + [True] * len(pl),
                    base_scale=image_size, scale=1024, is_line=[False] * len(room_polys) + [True] * len(pl))

                # gt_floorplan_map = plot_floorplan_with_regions(
                #     gt_room_polys, matching_labels=gt_mask, base_scale=image_size, scale=1024
                # )
                # floorplan_map = plot_floorplan_with_regions(
                #     room_polys, matching_labels=pred_mask, base_scale=image_size, scale=1024
                # )
                if not merge:
                    cv2.imwrite(os.path.join(output_dir, "{}_pred_floorplan.png".format(scene_ids[i])), floorplan_map)
                    cv2.imwrite(os.path.join(output_dir, "{}_gt_floorplan.png".format(scene_ids[i])), gt_floorplan_map)
                else:
                    concatenated_map = concat_floorplan_maps(gt_floorplan_map, floorplan_map, plot_statistics)
                    cv2.imwrite(
                        os.path.join(output_dir, "{}_pred_floorplan.png".format(scene_ids[i])), concatenated_map
                    )

                if semantic_rich:
                    _, ID2CLASS_LABEL = get_dataset_class_labels(dataset_name)
                    # add lines to the shapes to be predicted
                    shapes = [[p, t, False] for p, t in zip(room_polys + window_doors, room_types + window_doors_types)]
                    shapes += [[p, t, True] for p, t in pred_lines]
                    floorplan_map = plot_semantic_rich_floorplan_opencv(
                        zip(room_polys + window_doors, room_types + window_doors_types),
                        os.path.join(output_dir, "{}_pred_floorplan_sem.png".format(scene_ids[i])),
                        door_window_index=door_window_index,
                        semantics_label_mapping=ID2CLASS_LABEL,
                        img_w=image_size,
                        img_h=image_size,
                        scale=1,
                        plot_text=False,
                    )

            if save_pred:
                # Save room_polys as JSON
                json_path = os.path.join(output_dir, "jsons", "{}.json".format(str(scene_ids[i]).zfill(5)))
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                polys_list = [poly.astype(float).tolist() for poly in room_polys]
                if semantic_rich:
                    polys_list += [window_door.astype(float).tolist() for window_door in window_doors]
                    types_list = room_types + window_doors_types
                else:
                    types_list = [-1] * len(polys_list)

                output_json = [
                    {
                        "image_id": str(scene_ids[i]).zfill(5),
                        "segmentation": polys_list[instance_id],
                        "category_id": int(types_list[instance_id]),
                        "id": instance_id,
                    }
                    for instance_id in range(len(polys_list))
                ]
                with open(json_path, "w") as json_file:
                    json.dump(output_json, json_file)

                json_result_path = os.path.join(
                    output_dir, "result_jsons", "{}.json".format(str(scene_ids[i]).zfill(5))
                )
                new_quant_result_dict_scene = compute_f1(copy.deepcopy(quant_result_dict_scene), metric_category)
                os.makedirs(os.path.dirname(json_result_path), exist_ok=True)
                with open(json_result_path, "w") as json_file:
                    json.dump(new_quant_result_dict_scene, json_file)

            if plot_gt:
                gt_image = np.transpose(samples[i].cpu().numpy(), (1, 2, 0))
                gt_image = (gt_image * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, "{}_gt.png".format(scene_ids[i])), gt_image)

            if plot_density:
                pred_room_map = plot_density_map(samples[i], image_size, room_polys + pl,
                                                 list(pred_room_label_per_scene) + [t for _, t in pred_lines],
                                                 plot_text=False, is_line=[False] * len(room_polys) + [True] * len(pl))
                gt_room_map = plot_density_map(samples[i], image_size, gt_polys + gl,
                                               list(gt_polys_types) + [t for _, t in gt_lines],
                                               plot_text=False, is_line=[False] * len(gt_polys) + [True] * len(gl))
                                               
                # pred_room_map = plot_density_map(
                #     samples[i], image_size, room_polys, pred_room_label_per_scene, plot_text=False
                # )
                # gt_room_map = plot_density_map(samples[i], image_size, gt_polys, gt_polys_types, plot_text=False)

                if merge:
                    concatenated_map = concat_floorplan_maps(gt_room_map, pred_room_map, plot_statistics)
                    cv2.imwrite(
                        os.path.join(output_dir, "{}_pred_room_map.png".format(scene_ids[i])), concatenated_map
                    )
                else:
                    cv2.imwrite(os.path.join(output_dir, "{}_pred_room_map.png".format(scene_ids[i])), pred_room_map)
                    cv2.imwrite(os.path.join(output_dir, "{}_gt_room_map.png".format(scene_ids[i])), gt_room_map)

    for k in quant_result_dict.keys():
        quant_result_dict[k] /= float(scene_counter)
    quant_result_dict = compute_f1(quant_result_dict, metric_category)

    # add line results, only for labels that appear in images
    if line_results:
        # get all per-label metrics for the set of images in the sample
        keys = {k for r in line_results for k in r}
        # compute mean for each label across all images that have it
        quant_result_dict.update({k: float(np.mean([r[k] for r in line_results if k in r])) for k in keys})

    # add per-label results
    # each label averaged over the images that contain it
    if label_results:
        names, _ = get_dataset_class_labels(dataset_name)
        keys = sorted({k for r in label_results for k in r}, key=lambda k: (int(k.split("_")[1]), k))
        quant_result_dict["per_label"] = {
            f"{k.split('_')[0]}_{names[int(k.split('_')[1])]}": float(np.mean([r[k] for r in label_results if k in r]))
            for k in keys}
        for k, v in quant_result_dict["per_label"].items():
            print(f"{k:<25s} {v:.3f}")

    print("*************************************************")
    print(quant_result_dict)
    print("*************************************************")

    with open(os.path.join(output_dir, "results.txt"), "w") as file:
        file.write(json.dumps(quant_result_dict))


def generate(
    model,
    samples,
    semantic_rich=False,
    poly2seq: bool = False,
    use_cache=True,
    per_token_sem_loss=False,
    drop_wd=False,
):
    """
    Generate room polygons and labels from model predictions.

    This function supports two generation modes:
    - Non-poly2seq mode (poly2seq=False): RoomFormer generation with fg_mask filtering
    - Poly2seq mode (poly2seq=True): Raster2Seq generation with sequence-based predictions

    Args:
        model: The trained model to use for inference.
        samples: Input image samples (list of tensors).
        semantic_rich (bool, optional): If True, predict room types and window/door elements. Default is False.
        poly2seq (bool, optional): If True, uses sequence-based prediction mode with forward_inference.
                                   If False, uses mask-based filtering with fg_mask. Defaults to False.
        use_cache (bool, optional): If True, use caching in forward_inference. Only used when poly2seq=True. Default is True.
        per_token_sem_loss (bool, optional): If True, computes semantic class using voting across tokens.
                                             Only used when poly2seq=True. Default is False.
        drop_wd (bool, optional): If True, exclude window/door elements from output. Default is False.

    Returns:
        dict: Dictionary containing:
            - 'room': List of room polygons (and optionally window/door lines) per scene
            - 'labels': List of class labels per scene
    """

    model.eval()
    image_size = samples[0].size(2)

    # Get predictions based on mode
    if poly2seq:
        outputs = model.forward_inference(samples, use_cache)
        pred_corners = outputs["gen_out"]
        fg_mask = None
    else:
        outputs = model(samples)
        pred_corners = outputs["pred_coords"]
        pred_logits = outputs["pred_logits"]
        fg_mask = torch.sigmoid(pred_logits) > 0.5  # select valid corners

    bs = outputs["pred_logits"].shape[0]

    if "pred_room_logits" in outputs:
        pred_room_logits = outputs["pred_room_logits"]
        prob = torch.nn.functional.softmax(pred_room_logits, -1)
        _, pred_room_label = prob[..., :-1].max(-1)

    outputs = []
    output_classes = []

    # process per scene
    for i in range(bs):
        room_polys = []

        if semantic_rich:
            room_types = []
            window_doors = []
            window_doors_types = []
        else:
            window_doors = None
            room_types = None

        scene_outputs = _process_predictions(
            pred_corners,
            i,
            semantic_rich,
            poly2seq,
            fg_mask,
            image_size,
            pred_room_label if semantic_rich else None,
            pred_room_logits if semantic_rich else None,
            False,
            per_token_sem_loss,
            wd_as_line=True,
            door_window_index=None,
            dataset_name=None,
        )
        room_polys = scene_outputs["room_polys"]
        room_types = scene_outputs["room_types"]
        window_doors = scene_outputs["window_doors"]
        window_doors_types = scene_outputs["window_doors_types"]

        if not drop_wd and window_doors:
            outputs.append(room_polys + window_doors)
            output_classes.append(room_types + window_doors_types)
        else:
            outputs.append(room_polys)
            output_classes.append(room_types)

    out_dict = {"room": outputs, "labels": output_classes}
    return out_dict


def generate_with_refinement(
    model,
    samples,
    semantic_rich=False,
    poly2seq: bool = False,
    use_cache=True,
    per_token_sem_loss=False,
    drop_wd=False,
    threshold=0.5,
):
    """
    Generate room polygons and labels from model predictions with test-time refinement.
    """

    model.eval()
    image_size = samples[0].size(2)

    # Get predictions based on mode
    if poly2seq:
        outputs_ref, out1, good_out = model.forward_refinement(samples, use_cache, threshold=threshold)
        pred_corners = outputs_ref["gen_out"]
        fg_mask = None
    else:
        outputs_ref = model(samples)
        pred_corners = outputs_ref["pred_coords"]
        pred_logits = outputs_ref["pred_logits"]
        fg_mask = torch.sigmoid(pred_logits) > 0.5  # select valid corners
        out1, good_out = None, None

    bs = outputs_ref["pred_logits"].shape[0]

    if "pred_room_logits" in outputs_ref:
        pred_room_logits = outputs_ref["pred_room_logits"]
        prob = torch.nn.functional.softmax(pred_room_logits, -1)
        _, pred_room_label = prob[..., :-1].max(-1)

    outputs = []
    output_classes = []
    refinement_list = []
    refinement_polygon_indices = []

    # process per scene
    for i in range(bs):
        room_polys = []

        if semantic_rich:
            room_types = []
            window_doors = []
            window_doors_types = []
        else:
            window_doors = None
            room_types = None

        scene_outputs = _process_predictions(
            pred_corners,
            i,
            semantic_rich,
            poly2seq,
            fg_mask,
            image_size,
            pred_room_label if semantic_rich else None,
            pred_room_logits if semantic_rich else None,
            False,
            per_token_sem_loss,
            wd_as_line=True,
            door_window_index=None,
            dataset_name=None,
        )
        room_polys = scene_outputs["room_polys"]
        room_types = scene_outputs["room_types"]
        window_doors = scene_outputs["window_doors"]
        window_doors_types = scene_outputs["window_doors_types"]
        pred_room_label_per_scene = scene_outputs["pred_room_label_per_scene"]

        if not drop_wd and window_doors:
            outputs.append(room_polys + window_doors)
            output_classes.append(room_types + window_doors_types)
        else:
            outputs.append(room_polys)
            output_classes.append(room_types)

        # Debug plotting for refinement
        if poly2seq and out1[i] is not None:
            # Process Pass 1
            out1_i = out1[i]
            if "pred_room_logits" in out1_i:
                p1_prob = torch.nn.functional.softmax(out1_i["pred_room_logits"], -1)
                _, p1_labels = p1_prob[..., :-1].max(-1)
                p1_logits = out1_i["pred_room_logits"]
            else:
                p1_labels, p1_logits = None, None

            scene_out1 = _process_predictions(
                out1_i["gen_out"],
                0,
                semantic_rich,
                poly2seq,
                None,
                image_size,
                p1_labels,
                p1_logits,
                False,
                per_token_sem_loss,
                wd_as_line=True,
                door_window_index=None,
                dataset_name=None,
            )

            # Process Good Polys
            good_out_i = good_out[i]
            if "pred_room_logits" in good_out_i:
                good_prob = torch.nn.functional.softmax(good_out_i["pred_room_logits"], -1)
                _, good_labels = good_prob[..., :-1].max(-1)
                good_logits = good_out_i["pred_room_logits"]
            else:
                good_labels, good_logits = None, None

            scene_good = _process_predictions(
                good_out_i["gen_out"],
                0,
                semantic_rich,
                poly2seq,
                None,
                image_size,
                good_labels,
                good_logits,
                False,
                per_token_sem_loss,
                wd_as_line=True,
                door_window_index=None,
                dataset_name=None,
            )

            # # Plot Pass 1
            # # img1 = plot_density_map(
            # #     samples[i], image_size, scene_out1["room_polys"], scene_out1["pred_room_label_per_scene"], plot_text=False
            # # )
            # polys1 = scene_out1["room_polys"] + (scene_out1["window_doors"] if scene_out1["window_doors"] is not None else [])
            # labels1 = (scene_out1["room_types"] if scene_out1["room_types"] is not None else scene_out1["pred_room_label_per_scene"]) + \
            #     (scene_out1["window_doors_types"] if scene_out1["window_doors_types"] is not None else [])
            # img1 = plot_semantic_rich_floorplan_opencv(
            #     zip(polys1, labels1), None, img_w=image_size, img_h=image_size, plot_text=False, is_sem=semantic_rich
            # )
            # # cv2.imwrite(f"refinement_vis/sample_{i}_pass1.png", img1)
            # # Plot Good
            # # img_good = plot_density_map(
            # #     samples[i], image_size, scene_good["room_polys"], scene_good["pred_room_label_per_scene"], plot_text=False
            # # )
            # polys2 = scene_good["room_polys"] + (scene_good["window_doors"] if scene_good["window_doors"] is not None else [])
            # labels2 = (scene_good["room_types"] if scene_good["room_types"] is not None else scene_good["pred_room_label_per_scene"]) + \
            #     (scene_good["window_doors_types"] if scene_good["window_doors_types"] is not None else [])
            # img_good = plot_semantic_rich_floorplan_opencv(
            #     zip(polys2, labels2), None, img_w=image_size, img_h=image_size, plot_text=False, is_sem=semantic_rich
            # )
            # # cv2.imwrite(f"refinement_vis/sample_{i}_good.png", img_good)
            # # Plot Final
            # # img_final = plot_density_map(
            # #     samples[i], image_size, room_polys, pred_room_label_per_scene, plot_text=False
            # # )
            # polys_final = room_polys + (window_doors if window_doors is not None else [])
            # labels_final = (room_types if room_types is not None else pred_room_label_per_scene) + \
            #     (window_doors_types if window_doors_types is not None else []) 
            # img_final = plot_semantic_rich_floorplan_opencv(
            #     zip(polys_final, labels_final), None, img_w=image_size, img_h=image_size, plot_text=False, is_sem=semantic_rich
            # )
            # cv2.imwrite(f"refinement_vis/sample_{i}_final.png", img_final)
            refinement_list.append([scene_out1, scene_good, scene_outputs])

            # Compute which polygon indices in the final output are refined/new
            num_kept_rooms = len(scene_good["room_polys"])
            num_kept_wd = len(scene_good["window_doors"]) if scene_good["window_doors"] else 0
            total_rooms = len(room_polys)
            total_wd = len(window_doors) if window_doors else 0

            refined_indices = list(range(num_kept_rooms, total_rooms))
            if not drop_wd and window_doors:
                refined_indices += list(range(total_rooms + num_kept_wd, total_rooms + total_wd))

            refinement_polygon_indices.append(refined_indices)
        else:
            refinement_list.append([None, None, None])
            refinement_polygon_indices.append(None)

    out_dict = {"room": outputs,
                "labels": output_classes,
                "refinement_list": refinement_list,
                "refinement_polygon_indices": refinement_polygon_indices}
    return out_dict
