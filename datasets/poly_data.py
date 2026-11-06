import math
import os
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from datasets.data_utils import sort_polygons
from datasets.discrete_tokenizer import DiscreteTokenizer
from datasets.transforms import ResizeAndPad
from detectron2.data import transforms as T
from detectron2.data.detection_utils import annotations_to_instances, transform_instance_annotations
from detectron2.structures import BoxMode
from util.poly_ops import resort_corners


class TokenType(Enum):
    """0 for <coord>, 1 for <sep>, 2 for <eos>, 3 for <cls>"""

    coord = 0
    sep = 1
    eos = 2
    cls = 3


WD_INDEX = {
    "stru3d": [16, 17],
    "cubicasa": [9, 10],
    "waffle": [],
    "r2g": [],
}


class MultiPoly(Dataset):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        semantic_classes,
        dataset_name="",
        image_norm=False,
        poly2seq=False,
        converter_version="v1",
        random_drop_rate=0.0,
        **kwargs,
    ):
        super(MultiPoly, self).__init__()

        self.root = img_folder
        self._transforms = transforms
        self.semantic_classes = semantic_classes
        self.dataset_name = dataset_name

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.poly2seq = poly2seq
        self.prepare = ConvertToCocoDictWithOrder_plus(
            self.root,
            self._transforms,
            image_norm,
            poly2seq,
            semantic_classes=semantic_classes,
            order_type=["l2r", "r2l"][converter_version == "v3_flipped"],
            random_drop_rate=random_drop_rate,
            **kwargs,
        )

    def get_image(self, path):
        return Image.open(os.path.join(self.root, path))

    def get_vocab_size(self):
        if self.poly2seq:
            return len(self.prepare.tokenizer)
        return None

    def get_tokenizer(self):
        if self.poly2seq:
            return self.prepare.tokenizer
        return None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: COCO format dict
        """
        coco = self.coco
        img_id = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        ### Note: here is a hack which assumes door/window have category_id 16, 17 in structured3D
        if self.semantic_classes == -1:
            if self.dataset_name == "stru3d":
                target = [t for t in target if t["category_id"] not in WD_INDEX["stru3d"]]
            # elif self.dataset_name == 'rplan':
            #     target = [t for t in target if t['category_id'] not in [9, 11]]
            elif self.dataset_name == "cubicasa":
                target = [t for t in target if t["category_id"] not in WD_INDEX["cubicasa"]]

        path = coco.loadImgs(img_id)[0]["file_name"]

        record = self.prepare(img_id, path, target)

        return record


class MultiPolyWD(MultiPoly):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: COCO format dict
        """
        coco = self.coco
        img_id = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        ### Note: here is a hack which assumes door/window have category_id 16, 17 in structured3D
        # if self.semantic_classes == -1:
        #     if self.dataset_name == 'stru3d':
        #         target = [t for t in target if t['category_id'] not in [16, 17]]
        #     elif self.dataset_name == 'rplan':
        #         target = [t for t in target if t['category_id'] not in [9, 11]]
        #     elif self.dataset_name == 'cubicasa':
        #         target = [t for t in target if t['category_id'] not in [9, 10]]

        if self.dataset_name == "stru3d":
            target = [t for t in target if t["category_id"] in [16, 17]]
        elif self.dataset_name == "rplan":
            target = [t for t in target if t["category_id"] in [9, 11]]
        elif self.dataset_name == "cubicasa":
            target = [t for t in target if t["category_id"] in [9, 10]]

        path = coco.loadImgs(img_id)[0]["file_name"]
        record = self.prepare(img_id, path, target)

        return record


class ConvertToCocoDict(object):
    def __init__(
        self,
        root,
        augmentations,
        image_norm,
        poly2seq=False,
        semantic_classes=-1,
        add_cls_token=False,
        per_token_class=False,
        mask_format="polygon",
        **kwargs,
    ):
        self.root = root
        self.augmentations = augmentations
        if image_norm:
            self.image_normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            self.image_normalize = None

        self.semantic_classes = semantic_classes
        self.poly2seq = poly2seq
        if poly2seq:
            self.tokenizer = DiscreteTokenizer(add_cls=add_cls_token, **kwargs)
            self.add_cls_token = add_cls_token
        self.per_token_class = per_token_class
        self.mask_format = mask_format

    def _expand_image_dims(self, x):
        if len(x.shape) == 2:
            exp_img = np.expand_dims(x, 0)
        else:
            exp_img = x.transpose((2, 0, 1))  # (h,w,c) -> (c,h,w)
        return exp_img

    def __call__(self, img_id, path, target):

        file_name = os.path.join(self.root, path)

        img = np.array(Image.open(file_name))

        #### NEW
        if len(img.shape) >= 3:
            if img.shape[-1] > 3:  # drop alpha channel
                img = img[:, :, :3]
            w, h = img.shape[:-1]
        else:
            # print(img.shape, file_name)
            w, h = img.shape
        #### NEW

        record = {}
        record["file_name"] = file_name
        record["height"] = h
        record["width"] = w
        record["image_id"] = img_id

        for obj in target:
            obj["bbox_mode"] = BoxMode.XYWH_ABS

        record["annotations"] = target

        if self.augmentations is None:
            record["image"] = (1 / 255) * torch.as_tensor(np.ascontiguousarray(self._expand_image_dims(img)))
            record["instances"] = annotations_to_instances(target, (h, w), mask_format=self.mask_format)
        else:
            aug_input = T.AugInput(img)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            record["image"] = (1 / 255) * torch.as_tensor(np.array(self._expand_image_dims(image)))
            h, w = image.shape[:2]  # update size

            annos = [
                transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in record.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # resort corners after augmentation: so that all corners start from upper-left counterclockwise
            for anno in annos:
                anno["segmentation"][0] = resort_corners(anno["segmentation"][0])

            record["instances"] = annotations_to_instances(annos, (h, w), mask_format=self.mask_format)

        #### NEW ####
        if self.image_normalize is not None:
            record["image"] = self.image_normalize(record["image"])

        # convert polygons to sequences
        if self.poly2seq:
            # only happend for wdonly
            if not hasattr(record["instances"], "gt_masks"):
                polygons = [np.array([[0.0, 0.0]])]
                polygons_label = [self.semantic_classes - 1]  # dummy class
            else:
                polygons = [
                    np.clip(np.array(inst).reshape(-1, 2) / (w - 1), 0, 1)
                    for inst in record["instances"].gt_masks.polygons
                ]
                polygons_label = [inst.item() for inst in record["instances"].gt_classes]
            record.update(
                self._get_bilinear_interpolation_coeffs(
                    polygons, polygons_label, self.add_cls_token, self.per_token_class
                )
            )

        return record

    def _get_bilinear_interpolation_coeffs(self, polygons, polygons_label, add_cls_token=False, per_token_class=False):
        num_bins = self.tokenizer.num_bins
        quant_poly = [poly * (num_bins - 1) for poly in polygons]
        index11 = [[math.floor(p[0]) * num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
        index21 = [[math.ceil(p[0]) * num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
        index12 = [[math.floor(p[0]) * num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]
        index22 = [[math.ceil(p[0]) * num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]

        seq11 = self.tokenizer(index11, add_bos=True, add_eos=False, dtype=torch.long)
        seq21 = self.tokenizer(index21, add_bos=True, add_eos=False, dtype=torch.long)
        seq12 = self.tokenizer(index12, add_bos=True, add_eos=False, dtype=torch.long)
        seq22 = self.tokenizer(index22, add_bos=True, add_eos=False, dtype=torch.long)

        # in real values insteads
        target_seq = []
        token_labels = []  # 0 for <coord>, 1 for <sep>, 2 for <eos>, 3 for <cls>
        num_extra = 1 if not add_cls_token else 2  # cls and sep
        count_polys = 0
        for poly in polygons:
            cur_len = len(token_labels)
            if cur_len + len(poly) + num_extra > self.tokenizer.seq_len:
                break  # INFO: change from break to continue
            token_labels.extend([TokenType.coord.value] * len(poly))
            if add_cls_token:
                token_labels.append(TokenType.cls.value)  # cls token
            token_labels.append(TokenType.sep.value)  # separator token
            target_seq.extend(poly)
            if add_cls_token:
                target_seq.append([0, 0])  # padding for cls token
            target_seq.append([0, 0])  # padding for sep/end token
            count_polys += 1
        # remove last separator token
        if len(token_labels) > 0:
            token_labels[-1] = TokenType.eos.value
        mask = torch.ones(self.tokenizer.seq_len, dtype=torch.bool)
        if len(token_labels) < self.tokenizer.seq_len:
            mask[len(token_labels) :] = 0
        target_seq = self.tokenizer._padding(target_seq, [0, 0], dtype=torch.float32)
        token_labels = self.tokenizer._padding(token_labels, -1, dtype=torch.long)

        delta_x1 = [0]  # [0] for bos token
        for polygon in quant_poly[:count_polys]:
            delta = [poly_point[0] - math.floor(poly_point[0]) for poly_point in polygon]
            delta_x1.extend(delta)
            if add_cls_token:
                delta_x1.extend([0])  # for cls token
            delta_x1.extend([0])  # for separator token
        delta_x1 = delta_x1[:-1]  # there is no separator token in the end
        delta_x1 = self.tokenizer._padding(delta_x1, 0, dtype=torch.float32)
        delta_x2 = 1 - delta_x1

        delta_y1 = [0]  # [0] for bos token
        for polygon in quant_poly[:count_polys]:
            delta = [poly_point[1] - math.floor(poly_point[1]) for poly_point in polygon]
            delta_y1.extend(delta)
            if add_cls_token:
                delta_y1.extend([0])  # for cls token
            delta_y1.extend([0])  # for separator token
        delta_y1 = delta_y1[:-1]  # there is no separator token in the end
        delta_y1 = self.tokenizer._padding(delta_y1, 0, dtype=torch.float32)
        delta_y2 = 1 - delta_y1

        if not per_token_class:
            target_polygon_labels = polygons_label[:count_polys]
        else:
            target_polygon_labels = []
            for poly, poly_label in zip(quant_poly[:count_polys], polygons_label[:count_polys]):
                target_polygon_labels.extend([poly_label] * len(poly))
                target_polygon_labels.append(self.semantic_classes - 1)  # undefined class for <sep> and <eos> token

        max_label_length = self.tokenizer.seq_len
        if len(polygons_label) < max_label_length:
            target_polygon_labels.extend([-1] * (max_label_length - len(target_polygon_labels)))

        target_polygon_labels = torch.tensor(target_polygon_labels, dtype=torch.long)

        return {
            "delta_x1": delta_x1,
            "delta_x2": delta_x2,
            "delta_y1": delta_y1,
            "delta_y2": delta_y2,
            "seq11": seq11,
            "seq21": seq21,
            "seq12": seq12,
            "seq22": seq22,
            "target_seq": target_seq,
            "token_labels": token_labels,
            "mask": mask,
            "target_polygon_labels": target_polygon_labels,
        }


class ConvertToCocoDictWithOrder_plus(ConvertToCocoDict):
    def __init__(
        self,
        root,
        augmentations,
        image_norm,
        poly2seq=False,
        semantic_classes=-1,
        add_cls_token=False,
        per_token_class=False,
        mask_format="polygon",
        dataset_name="stru3d",
        order_type="l2r",
        random_drop_rate=0.0,
        row_tolerance=None,
        **kwargs,
    ):
        super().__init__(
            root,
            augmentations,
            image_norm,
            poly2seq,
            semantic_classes,
            add_cls_token,
            per_token_class,
            mask_format,
            **kwargs,
        )
        self.dataset_name = dataset_name
        self.order_type = order_type  # l2r, r2l
        self.random_drop_rate = random_drop_rate
        # Shape order: sort_polygons groups shapes into rows whose top corners are within `tolerance`, then goes
        # left to right. The default tolerance (20) is applied to 0-1 coordinates, which puts every shape in
        # one row (pure left-to-right order). row_tolerance sets the row height as a fraction of the image
        # (e.g. 0.08), giving reading order (rows top to bottom) at any image size; None keeps the original.
        self.row_tolerance = row_tolerance
        self.tokenizer = DiscreteTokenizer(add_cls=add_cls_token, **kwargs)

    def _get_bilinear_interpolation_coeffs(self, polygons, polygons_label, add_cls_token=False, per_token_class=False):
        num_bins = self.tokenizer.num_bins
        room_indices = [
            poly_idx
            for poly_idx, poly_label in enumerate(polygons_label)
            if poly_label not in WD_INDEX[self.dataset_name]
        ]
        wd_indices = [
            poly_idx for poly_idx, poly_label in enumerate(polygons_label) if poly_label in WD_INDEX[self.dataset_name]
        ]

        tol = {} if self.row_tolerance is None else {"tolerance": self.row_tolerance}   # polygons are in 0-1 units
        _, room_sorted_indices = sort_polygons(
            [polygons[poly_idx] for poly_idx in room_indices], reverse=(self.order_type == "r2l"), **tol
        )
        _, wd_sorted_indices = sort_polygons(
            [polygons[poly_idx] for poly_idx in wd_indices], reverse=(self.order_type == "r2l"), **tol
        )
        room_indices = [room_indices[_idx] for _idx in room_sorted_indices]
        wd_indices = [wd_indices[_idx] for _idx in wd_sorted_indices]

        #### NEW ####
        combined_indices = room_indices + wd_indices  # room first
        if self.random_drop_rate > 0 and len(combined_indices) > 2:
            keep_indices = np.where(np.random.rand(len(combined_indices)) >= self.random_drop_rate)[0].tolist()
            if len(keep_indices) > 0:  # Only apply drop if we have something left
                combined_indices = [combined_indices[i] for i in keep_indices]
        #### NEW ####

        polygons = [polygons[i] for i in combined_indices]
        polygons_label = [polygons_label[i] for i in combined_indices]

        quant_poly = [poly * (num_bins - 1) for poly in polygons]
        index11 = [[math.floor(p[0]) * num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
        index21 = [[math.ceil(p[0]) * num_bins + math.floor(p[1]) for p in poly] for poly in quant_poly]
        index12 = [[math.floor(p[0]) * num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]
        index22 = [[math.ceil(p[0]) * num_bins + math.ceil(p[1]) for p in poly] for poly in quant_poly]

        seq11 = self.tokenizer(index11, add_bos=True, add_eos=False, dtype=torch.long)
        seq21 = self.tokenizer(index21, add_bos=True, add_eos=False, dtype=torch.long)
        seq12 = self.tokenizer(index12, add_bos=True, add_eos=False, dtype=torch.long)
        seq22, poly_indices = self.tokenizer(
            index22, add_bos=True, add_eos=False, dtype=torch.long, return_indices=True
        )

        # in real values insteads
        target_seq = []
        token_labels = []  # 0 for <coord>, 1 for <sep>, 2 for <eos>, 3 for <cls>

        for i in poly_indices:
            token_labels.extend([TokenType.coord.value] * len(polygons[i]))
            if add_cls_token:
                token_labels.append(TokenType.cls.value)  # cls token
            token_labels.append(TokenType.sep.value)  # separator token
            target_seq.extend(polygons[i])
            if add_cls_token:
                target_seq.append([0, 0])  # padding for cls token
            target_seq.append([0, 0])  # padding for sep/end token
        # remove last separator token
        token_labels[-1] = TokenType.eos.value

        mask = torch.ones(self.tokenizer.seq_len, dtype=torch.bool)
        if len(token_labels) < self.tokenizer.seq_len:
            mask[len(token_labels) :] = 0
        target_seq = self.tokenizer._padding(target_seq, [0, 0], dtype=torch.float32)
        token_labels = self.tokenizer._padding(token_labels, -1, dtype=torch.long)

        delta_x1 = [0]  # [0] for bos token
        for i in poly_indices:
            polygon = quant_poly[i]
            delta = [poly_point[0] - math.floor(poly_point[0]) for poly_point in polygon]
            delta_x1.extend(delta)
            if add_cls_token:
                delta_x1.extend([0])  # for cls token
            delta_x1.extend([0])  # for separator token
        delta_x1 = delta_x1[:-1]  # there is no separator token in the end
        delta_x1 = self.tokenizer._padding(delta_x1, 0, dtype=torch.float32)
        delta_x2 = 1 - delta_x1

        delta_y1 = [0]  # [0] for bos token
        for i in poly_indices:
            polygon = quant_poly[i]
            delta = [poly_point[1] - math.floor(poly_point[1]) for poly_point in polygon]
            delta_y1.extend(delta)
            if add_cls_token:
                delta_y1.extend([0])  # for cls token
            delta_y1.extend([0])  # for separator token
        delta_y1 = delta_y1[:-1]  # there is no separator token in the end
        delta_y1 = self.tokenizer._padding(delta_y1, 0, dtype=torch.float32)
        delta_y2 = 1 - delta_y1

        if not per_token_class:
            target_polygon_labels = [polygons_label[i] for i in poly_indices]  # polygons_label[:count_polys]
            input_polygon_labels = torch.tensor(target_polygon_labels.copy(), dtype=torch.long)
        else:
            target_polygon_labels = []
            for i in poly_indices:
                poly, poly_label = quant_poly[i], polygons_label[i]
                target_polygon_labels.extend([poly_label] * len(poly))
                target_polygon_labels.append(self.semantic_classes - 1)  # undefined class for <sep> and <eos> token
            input_polygon_labels = torch.tensor(
                [self.semantic_classes - 1] + target_polygon_labels.copy()[:-1], dtype=torch.long
            )  # right shift by one: <bos>, ..., <coord>

        max_label_length = self.tokenizer.seq_len
        if len(polygons_label) < max_label_length:
            target_polygon_labels.extend([-1] * (max_label_length - len(target_polygon_labels)))

        target_polygon_labels = torch.tensor(target_polygon_labels, dtype=torch.long)

        return {
            "delta_x1": delta_x1,
            "delta_x2": delta_x2,
            "delta_y1": delta_y1,
            "delta_y2": delta_y2,
            "seq11": seq11,
            "seq21": seq21,
            "seq12": seq12,
            "seq22": seq22,
            "target_seq": target_seq,
            "token_labels": token_labels,
            "mask": mask,
            "target_polygon_labels": target_polygon_labels,
            "input_polygon_labels": input_polygon_labels,
        }


def make_poly_transforms(dataset_name, image_set, image_size=256, disable_image_transform=False):

    trans_list = []
    if dataset_name in ["cubicasa", "waffle"] or (dataset_name == "r2g" and image_size != 512):
        trans_list = [ResizeAndPad((image_size, image_size), pad_value=255)]

    if image_set == "train":
        if not disable_image_transform:
            trans_list.extend(
                [
                    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                    T.RandomRotation([0.0, 90.0, 180.0, 270.0], expand=False, center=None, sample_style="choice"),
                ]
            )
        return T.AugmentationList(trans_list)

    if image_set == "val" or image_set == "test":
        return None if len(trans_list) == 0 else T.AugmentationList(trans_list)

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.dataset_root)
    assert root.exists(), f"provided data path {root} does not exist"

    PATHS = {
        "train": (root / "train", root / "annotations" / "train.json"),
        "val": (root / "val", root / "annotations" / "val.json"),
        "test": (root / "test", root / "annotations" / "test.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    image_transform = make_poly_transforms(
        args.dataset_name,
        image_set,
        image_size=args.image_size,
        disable_image_transform=getattr(args, "disable_image_transform", False),
    )

    if args.wd_only:
        dataset = MultiPolyWD(
            img_folder,
            ann_file,
            transforms=image_transform,
            semantic_classes=args.semantic_classes,
            dataset_name=args.dataset_name,
            image_norm=args.image_norm,
            poly2seq=args.poly2seq,
            num_bins=args.num_bins,
            seq_len=args.seq_len,
            add_cls_token=args.add_cls_token,
            per_token_class=args.per_token_sem_loss,
            mask_format=getattr(args, "mask_format", "polygon"),
        )
    else:
        dataset = MultiPoly(
            img_folder,
            ann_file,
            transforms=image_transform,
            semantic_classes=args.semantic_classes,
            dataset_name=args.dataset_name,
            image_norm=args.image_norm,
            poly2seq=args.poly2seq,
            num_bins=args.num_bins,
            seq_len=args.seq_len,
            add_cls_token=args.add_cls_token,
            per_token_class=args.per_token_sem_loss,
            mask_format=getattr(args, "mask_format", "polygon"),
            converter_version=getattr(args, "converter_version", "v1"),
            random_drop_rate=getattr(args, "random_drop_rate", 0.0),
            row_tolerance=getattr(args, "row_tolerance", None),
        )

    return dataset
