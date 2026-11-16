"""
Utilities for polygon manipulation.
"""

import numpy as np
import torch


def is_clockwise(points):
    """Check whether a sequence of points is clockwise ordered"""
    # points is a list of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0


def orient_line(points):
    """Fixed direction for an open line (the line counterpart of resort_corners): keep the point
    order, but start at the end nearer the image's upper-left corner (smaller x^2 + y^2), so a line
    has one canonical sequence instead of two. Returns a flat array, like resort_corners."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if (pts[-1] ** 2).sum() < (pts[0] ** 2).sum():
        pts = pts[::-1]
    return pts.reshape(-1)


def resort_corners(corners):
    """Resort a sequence of corners so that the first corner starts
    from upper-left and counterclockwise ordered in image
    """
    corners = corners.reshape(-1, 2)
    x_y_square_sum = corners[:, 0] ** 2 + corners[:, 1] ** 2
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    ## sort points clockwise (counterclockwise in image)
    if not is_clockwise(corners_sorted[:, :2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners_sorted.reshape(-1)


def get_all_order_corners(corners):
    """Get all possible permutation of a polygon"""
    length = int(len(corners) / 2)
    all_corners = torch.stack([corners.roll(i * 2) for i in range(length)])
    return all_corners


def pad_gt_polys(gt_instances, num_queries_per_poly, image_size, drop_rate, device=None):
    """Pad the ground truth polygons so that they have a uniform length"""

    thr_length = num_queries_per_poly * 2
    room_targets = []
    # padding ground truth on-fly
    for gt_inst in gt_instances:
        room_dict = {}
        room_corners = []
        corner_labels = []
        corner_lengths = []

        for i, poly in enumerate(gt_inst.gt_masks.polygons):
            corners = torch.from_numpy(poly[0]).to(device)
            corners = torch.clip(corners, 0, image_size - 1) / (image_size - 1)

            # automatically skip the polygon if it is too long
            if len(corners) > thr_length:
                continue

            corner_lengths.append(len(corners))

            corners_pad = torch.zeros(num_queries_per_poly * 2, device=device)
            corners_pad[: len(corners)] = corners

            labels = torch.ones(int(len(corners) / 2), dtype=torch.int64).to(device)
            labels_pad = torch.zeros(num_queries_per_poly, device=device)
            labels_pad[: len(labels)] = labels
            room_corners.append(corners_pad)
            corner_labels.append(labels_pad)

        room_classes = gt_inst.gt_classes
        if drop_rate > 0.0:
            keep_indices = np.where(np.random.rand(len(room_corners)) >= drop_rate)[0].tolist()
            if len(keep_indices) > 0:  # Only apply drop if we have something left
                room_corners = [room_corners[i] for i in keep_indices]
                corner_labels = [corner_labels[i] for i in keep_indices]
                corner_lengths = [corner_lengths[i] for i in keep_indices]
                room_classes = gt_inst.gt_classes[keep_indices]

        room_dict = {
            "coords": torch.stack(room_corners).to(device),
            "labels": torch.stack(corner_labels).to(device),
            "lengths": torch.tensor(corner_lengths, device=device),
            "room_labels": room_classes.to(device),
        }
        room_targets.append(room_dict)

    return room_targets
