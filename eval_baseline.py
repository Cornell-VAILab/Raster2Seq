"""Score the colour-threshold baseline with Raster2Seq's own loader and metrics,
using the same rules as engine.py on the eval-osm-fixes branch:
  - ground truth from build_dataset (resize/pad to 512, seq_len truncation, as in training)
  - predicted polygons under --min_poly_area px^2 dropped, as engine.py does; with
    --filter_gt_by_area the same threshold also drops ground-truth polygons (lines exempt)
  - polygon metrics (room_*, corner_*, angles_*) on non-line shapes only
  - line metrics (line_*) from evaluations/line_eval.py, lines buffered --line_radius px

The colour-threshold baseline itself is predict() from baselines/color_threshold.py in the
Vectorizing-Raster-Maps repo (unchanged). It runs on the original full-size crop, and its points are
mapped into the same 512 px resized/padded frame as the ground truth.

Put this file in the root of your Raster2Seq folder and run from there, e.g.:
  python eval_baseline.py --dataset_root data/<coco_folder> --vrm_repo ../Vectorizing-Raster-Maps --num_images 20
Label ids, line classes and --semantic_classes are read from the dataset's annotations json.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluations"))
from datasets import build_dataset  # noqa: E402
from rplan_eval.Evaluator import Evaluator_RPlan  # noqa: E402
from line_eval import line_scores  # noqa: E402

try:  # per-label scores, in Jennah's line_eval.py
    from line_eval import label_scores  # noqa: E402
except ImportError:
    label_scores = None


LINE_NAMES = {"highway", "street", "railway", "path", "waterway", "runway", "unknown_line"}


def run_baseline(predict, file_name, name_to_id, image_size, skipped):
    """Colour-threshold shapes for one crop, as (Nx2 points in the 512 px frame, category_id).
    predict() runs on the original crop; points get the same resize + centred pad as ResizeAndPad."""
    arr = np.array(Image.open(file_name).convert("RGB"))
    h, w = arr.shape[:2]
    scale = min(image_size / h, image_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    top, left = (image_size - new_h) // 2, (image_size - new_w) // 2
    out = []
    for s in predict(arr):
        cid = name_to_id.get(s["label"])
        if cid is None:  # label the dataset vocabulary doesn't have
            skipped[s["label"]] += 1
            continue
        pts = np.asarray(s["points"], dtype=float).reshape(-1, 2)
        pts = pts * [new_w / w, new_h / h] + [left, top]
        out.append((pts, cid))
    return out


def keep_shape(poly, cls, line_classes, min_area):
    """Same size rule as engine.py: lines always kept, polygons need area >= min_area."""
    if cls in line_classes:
        return len(poly) >= 2
    return len(poly) >= 3 and Polygon(poly).area >= min_area


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--eval_set", default="test")
    p.add_argument("--dataset_name", default="osm")  # what Jennah trains with; "waffle" behaves the same
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_bins", type=int, default=32)
    p.add_argument("--vrm_repo", required=True, help="path to your Vectorizing-Raster-Maps folder")
    p.add_argument("--semantic_classes", type=int, default=None, help="default: number of labels + 1")
    p.add_argument("--line_classes", type=int, nargs="*", default=None, help="default: ids of line labels")
    p.add_argument("--line_radius", type=float, default=3)
    p.add_argument("--min_poly_area", type=float, default=100)
    p.add_argument("--filter_gt_by_area", action="store_true",
                   help="also drop ground-truth polygons under --min_poly_area (engine.FILTER_GT_BY_AREA)")
    p.add_argument("--per_label", action="store_true", help="also report precision / recall per label")
    p.add_argument("--num_images", type=int, default=-1)
    p.add_argument("--out", default="baseline_results.json")
    a = p.parse_args()

    sys.path.insert(0, os.path.abspath(a.vrm_repo))
    from baselines.color_threshold import predict  # noqa: E402

    with open(os.path.join(a.dataset_root, "annotations", f"{a.eval_set}.json")) as f:
        cats = json.load(f)["categories"]
    name_to_id = {c["name"]: c["id"] for c in cats}
    if a.semantic_classes is None:
        a.semantic_classes = len(cats) + 1
    if a.line_classes is None:
        a.line_classes = sorted(c["id"] for c in cats if c["name"] in LINE_NAMES)
    line_classes = set(a.line_classes)
    print(f"{len(cats)} labels, semantic_classes={a.semantic_classes}, line classes="
          f"{[(c['id'], c['name']) for c in cats if c['id'] in line_classes]}")

    # Same flags build_dataset reads from main_ddp.py / eval.py.
    args = argparse.Namespace(
        dataset_root=a.dataset_root, dataset_name=a.dataset_name, image_size=a.image_size,
        seq_len=a.seq_len, num_bins=a.num_bins, semantic_classes=a.semantic_classes,
        poly2seq=True, image_norm=False, wd_only=False, add_cls_token=False,
        per_token_sem_loss=True, converter_version="v3", disable_image_transform=True,
    )
    dataset = build_dataset(a.eval_set, args)
    evaluator = Evaluator_RPlan(disable_overlapping_filter=True, wd_as_line=False)

    totals, counts, skipped = defaultdict(float), defaultdict(int), defaultdict(int)
    per_label, per_label_n = defaultdict(float), defaultdict(int)
    n_images = len(dataset) if a.num_images < 0 else min(a.num_images, len(dataset))
    for idx in range(n_images):
        rec = dataset[idx]
        size = a.image_size

        inst = rec["instances"]
        gt = [(np.round(g[0].reshape(-1, 2)).astype(np.int32), int(c))
              for g, c in zip(inst.gt_masks.polygons, inst.gt_classes.tolist())]
        preds = [(np.round(np.asarray(q, dtype=float).reshape(-1, 2)).astype(np.int32), int(c))
                 for q, c in run_baseline(predict, rec["file_name"], name_to_id, size, skipped)]

        if a.filter_gt_by_area:
            gt = [(q, c) for q, c in gt if keep_shape(q, c, line_classes, a.min_poly_area)]
        preds = [(q, c) for q, c in preds if keep_shape(q, c, line_classes, a.min_poly_area)]

        # Lines: scored separately ({} when the crop has no true lines).
        if line_classes:
            lr = line_scores([q for q, _ in preds], [c for _, c in preds],
                             [q for q, _ in gt], [c for _, c in gt],
                             line_classes, img_size=(size, size), radius=a.line_radius)
            for k, v in lr.items():
                totals[k] += v
                counts[k] += 1

        if a.per_label and label_scores is not None and gt:
            ls = label_scores([q for q, _ in preds], [c for _, c in preds], [q for q, _ in gt], [c for _, c in gt],
                              line_classes, img_size=(size, size), radius=a.line_radius)
            for k, v in ls.items():
                per_label[k] += v
                per_label_n[k] += 1

        # Polygons: everything that is not a line.
        pp = [(q, c) for q, c in preds if c not in line_classes]
        gp = [(q, c) for q, c in gt if c not in line_classes]
        r = evaluator.evaluate_scene(room_polys=[q for q, _ in pp], room_types=[c for _, c in pp],
                                     gt_polys=[q for q, _ in gp], gt_polys_types=[c for _, c in gp],
                                     img_size=(size, size))
        for k in ("room_prec", "room_rec", "room_sem_prec", "room_sem_rec",
                  "corner_prec", "corner_rec", "angles_prec", "angles_rec"):
            totals[k] += r[k]
            counts[k] += 1

        if (idx + 1) % 50 == 0:
            print(f"{idx + 1}/{n_images} images")

    # Averaged per image, like engine.evaluate's metric_logger.
    res = {k: totals[k] / counts[k] for k in totals}
    for pre in ("room", "room_sem", "line", "line_sem", "corner", "angles"):
        pk, rk = f"{pre}_prec", f"{pre}_rec"
        if pk in res and res[pk] + res[rk] > 0:
            res[f"{pre}_f1"] = 2 * res[pk] * res[rk] / (res[pk] + res[rk])
    res["num_images"] = n_images
    if skipped:
        print(f"predicted labels not in the dataset (skipped): {dict(skipped)}")
    for k, v in res.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    labels = {}
    if per_label:
        id_to_name = {c["id"]: c["name"] for c in cats}
        for k in sorted(per_label):
            kind, cid = k.split("_")
            name = id_to_name.get(int(cid), cid)
            labels.setdefault(name, {"id": int(cid), "images": per_label_n[k]})[kind] = per_label[k] / per_label_n[k]
        print("\nper label (averaged over images that contain the label):")
        for name, d in sorted(labels.items(), key=lambda x: x[1]["id"]):
            print(f"  {d['id']:>2} {name:<14} prec {d.get('prec', 0):.3f}  rec {d.get('rec', 0):.3f}  ({d['images']} images)")
    with open(a.out, "w") as f:
        json.dump({"settings": vars(a), "results": res, "per_label": labels}, f, indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
