import numpy as np
from shapely.geometry import LineString

from evaluations.rplan_eval.Evaluator import Evaluator_RPlan, corner_metric_thresh


def buffer_lines(polys, types, line_classes, radius=2.0):
    """Lines (types in line_classes) widened `radius` px on each side into polygons (flat ends, mitred corners,
    like create_coco_osm.py --line_buffer_px); other shapes pass through unchanged."""
    out = []
    for p, t in zip(polys, types):
        # only buffer if the type is in one of the line classes
        # otherwise don't buffer and just append
        if t in line_classes and len(p) >= 2:
            g = LineString(p).buffer(radius, cap_style=2, join_style=2)
            if g.geom_type == "MultiPolygon":  
                # a line folding back on itself can split into pieces: keep the largest
                g = max(g.geoms, key=lambda x: x.area)
            if not g.is_empty:
                p = np.round(np.asarray(g.exterior.coords[:-1])).astype(np.int32)
        out.append(p)
    return out


def _scoring_helper(pred_polys, pred_types, gt_polys, gt_types, keep, line_classes, img_size, radius):
    pred = [(p, t) for p, t in zip(pred_polys, pred_types) if keep(t)]
    gt = [(p, t) for p, t in zip(gt_polys, gt_types) if keep(t)]
    if not gt:
        return None
    pt, gtt = [t for _, t in pred], [t for _, t in gt]
    return Evaluator_RPlan(disable_overlapping_filter=True, wd_as_line=False).evaluate_scene(
        room_polys=buffer_lines([p for p, _ in pred], pt, line_classes, radius), room_types=pt,
        gt_polys=buffer_lines([p for p, _ in gt], gtt, line_classes, radius), gt_polys_types=gtt, img_size=img_size)

def line_scores(pred_polys, pred_types, gt_polys, gt_types, line_classes, img_size, radius=corner_metric_thresh):
    """Line precision / recall: the lines are buffered `radius` px on each side (default: the 10 px corner
    tolerance) and matched like polygons (Evaluator_RPlan, IoU > 0.5); line_sem_* also need the right label.
    Returns {} without true lines."""
    r = _scoring_helper(pred_polys, pred_types, gt_polys, gt_types, lambda t: t in line_classes, line_classes, img_size, radius)
    if r is None:
        return {}
    # duplicates metrics but for lines
    return {"line_prec": r["room_prec"], "line_rec": r["room_rec"],
            "line_sem_prec": r["room_sem_prec"], "line_sem_rec": r["room_sem_rec"]}

def label_scores(pred_polys, pred_types, gt_polys, gt_types, line_classes, img_size, radius=corner_metric_thresh):
    """Per-label precision / recall: Evaluator_RPlan on one label's shapes at a time (lines buffered first).
    Keys: prec_<label id>, rec_<label id>, for labels with true shapes in the image."""
    out = {}
    for c in sorted(set(gt_types)):
        r = _scoring_helper(pred_polys, pred_types, gt_polys, gt_types, lambda t: t == c, line_classes, img_size, radius)
        out[f"prec_{c}"], out[f"rec_{c}"] = r["room_prec"], r["room_rec"]
    return out
