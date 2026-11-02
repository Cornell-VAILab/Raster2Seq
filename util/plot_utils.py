"""
Utilities for floorplan visualization.
"""

import math

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from descartes.patch import PolygonPatch
from imageio import imsave
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from PIL import ImageColor
from plotly.colors import qualitative
from shapely.geometry import LineString, Polygon

colors_12 = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58230",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd7b4",
]

semantics_cmap = {
    0: "#e6194b",
    1: "#3cb44b",
    2: "#ffe119",
    3: "#0082c8",
    4: "#f58230",
    5: "#911eb4",
    6: "#46f0f0",
    7: "#f032e6",
    8: "#d2f53c",
    9: "#fabebe",
    10: "#008080",
    11: "#e6beff",
    12: "#aa6e28",
    13: "#fffac8",
    14: "#800000",
    15: "#aaffc3",
    16: "#808000",
    17: "#ffd7b4",
}

S3D_LABEL = {
    0: "Living Room",
    1: "Kitchen",
    2: "Bedroom",
    3: "Bathroom",
    4: "Balcony",
    5: "Corridor",
    6: "Dining room",
    7: "Study",
    8: "Studio",
    9: "Store room",
    10: "Garden",
    11: "Laundry room",
    12: "Office",
    13: "Basement",
    14: "Garage",
    15: "Misc.",
    16: "Door",
    17: "Window",
}

CC5K_LABEL = {
    0: "Outdoor",
    1: "Kitchen",
    2: "Living Room",
    3: "Bed Room",
    4: "Bath",
    5: "Entry",
    6: "Storage",
    7: "Garage",
    8: "Undefined",
    9: "Window",
    10: "Door",
}

R2G_LABEL = {
    0: "unknown",
    1: "living_room",
    2: "kitchen",
    3: "bedroom",
    4: "bathroom",
    5: "restroom",
    6: "balcony",
    7: "closet",
    8: "corridor",
    9: "washing_room",
    10: "PS",
    11: "outside",
}

BLUE = "#6699cc"
GRAY = "#999999"
DARKGRAY = "#333333"
YELLOW = "#ffcc33"
GREEN = "#339933"
RED = "#ff3333"
BLACK = "#000000"


def auto_crop_whitespace(image, color_invert=True):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert the image so floorplan is white and background is black
    if color_invert:
        _, binary = cv2.threshold(255 - gray, 1, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find non-zero (non-white) content
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop image
    cropped_image = image[y : y + h, x : x + w].copy()

    # if polygons is None:
    #     return cropped_image, None

    # # Shift polygon coordinates
    # shifted_polygons = [
    #     [(px - x, py - y) for (px, py) in poly]
    #     for poly in polygons
    # ]
    return cropped_image, [x, y, w, h]  # shifted_polygons


def plot_floorplan_with_regions(
    regions,
    corners=None,
    edges=None,
    base_scale=256,
    scale=256,
    matching_labels=None,
    regions_type=None,
    plot_text=False,
    semantics_label_mapping=None,
):
    """Draw floorplan map where different colors indicate different rooms"""
    # cmap = get_cmap('tab20', 20) # nipy_spectral
    # colors = [cmap(x) for x in np.linspace(0, 1, 21)] # colors = colors_12
    colors = list(qualitative.Set3) + list(qualitative.Dark2)  # qualitative.Light24
    rgb_string_to_tuple = lambda rgb_string: tuple(float(x) / 255 for x in rgb_string.strip("rgb()").split(","))
    colors = [rgb_string_to_tuple(x) for x in colors]
    # colors = [to_rgb(x) for x in colors]
    gray_color = tuple(c / 255.0 for c in (255, 255, 255, 255))

    regions = [(region * scale / base_scale).round().astype(np.int32) for region in regions]

    # Ensure room_colors contains valid hex strings
    if matching_labels is None:
        room_colors = [to_hex(colors[i % len(colors)]) for i in range(len(regions))]
    else:
        room_colors = [
            to_hex(colors[i % len(colors)]) if matching_labels[i] else to_hex(gray_color[:3])
            for i in range(len(regions))
        ]

    # colorMap = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in room_colors]
    # colorMap = np.asarray(colorMap)
    colorMap = np.array([ImageColor.getrgb(h) for h in room_colors], dtype=np.uint8)
    if len(regions) > 0:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap], axis=0).astype(np.uint8)
    else:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0)], axis=0).astype(np.uint8)
    # when using opencv, we need to flip, from RGB to BGR
    colorMap = colorMap[:, ::-1]

    alpha_channels = np.zeros(colorMap.shape[0], dtype=np.uint8)
    alpha_channels[1 : len(regions) + 1] = 150

    colorMap = np.concatenate([colorMap, np.expand_dims(alpha_channels, axis=-1)], axis=-1)

    room_map = np.zeros([scale, scale]).astype(np.int32)
    # # sort regions
    # if len(regions) > 1:
    #     avg_corner = [region.mean(axis=0) for region in regions]
    #     ind = np.argsort(np.square(np.array(avg_corner)).sum(axis=1), axis=0)
    #     regions = [regions[_idx] for _idx in ind] # np.array(regions)[ind]

    for idx, polygon in enumerate(regions):
        cv2.fillPoly(room_map, [polygon], color=idx + 1)

    image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 4))

    pointColor = (0, 0, 0, 255)
    lineColor = (0, 0, 0, 255)

    for region in regions:
        for i, point in enumerate(region):
            if i == len(region) - 1:
                cv2.line(image, tuple(point), tuple(region[0]), color=lineColor, thickness=5)
            else:
                cv2.line(image, tuple(point), tuple(region[i + 1]), color=lineColor, thickness=5)

    for region in regions:
        for i, point in enumerate(region):
            cv2.circle(image, tuple(point), color=pointColor, radius=12, thickness=-1)
            cv2.circle(image, tuple(point), color=(255, 255, 255, 0), radius=6, thickness=-1)

    if plot_text:
        font_scale = 1.0
        text_padding = 1
        # Add room labels
        for points, poly_type in zip(regions, regions_type):
            # Calculate the centroid for text placement
            M = cv2.moments(points)
            if M["m00"] != 0:  # Avoid division by zero
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])

                # Get room label
                label = semantics_label_mapping[poly_type]

                # Get text size for centering and background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

                # Calculate text background rectangle
                text_x = centroid_x - text_size[0] // 2
                text_y = centroid_y + text_size[1] // 2

                # Create background for text
                rect_top_left = (text_x - text_padding, text_y - text_size[1] - text_padding)
                rect_bottom_right = (text_x + text_size[0] + text_padding, text_y + text_padding)

                # Draw semi-transparent white background for text
                background = image.copy()
                cv2.rectangle(background, rect_top_left, rect_bottom_right, (255, 255, 255), -1)

                # Blend the background
                cv2.addWeighted(background, 0.4, image, 0.6, 0, image)

                # cv2.rectangle(image, rect_top_left, rect_bottom_right,
                #             (255, 255, 255), -1)

                # Draw the text
                cv2.putText(
                    image,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),  # Black text
                    1,  # Thickness
                    cv2.LINE_AA,  # Anti-aliased text
                )

    return image


def plot_score_map(corner_map, scores):
    """Draw score map overlaid on the density map"""
    score_map = np.zeros([356, 356, 3])
    score_map[100:, 50:306] = corner_map
    cv2.putText(
        score_map,
        "room_prec: " + str(round(scores["room_prec"] * 100, 1)),
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (252, 252, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        score_map,
        "room_rec: " + str(round(scores["room_rec"] * 100, 1)),
        (190, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (252, 252, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        score_map,
        "corner_prec: " + str(round(scores["corner_prec"] * 100, 1)),
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        score_map,
        "corner_rec: " + str(round(scores["corner_rec"] * 100, 1)),
        (190, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        score_map,
        "angles_prec: " + str(round(scores["angles_prec"] * 100, 1)),
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        score_map,
        "angles_rec: " + str(round(scores["angles_rec"] * 100, 1)),
        (190, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return score_map


def plot_room_map(preds, room_map, room_id=0, im_size=256, plot_text=True, closed=True):
    """Draw room polygons overlaid on the density map (closed=False: an open line, no closing edge)"""
    centroid_x = int(np.mean(preds[:, 0]))
    centroid_y = int(np.mean(preds[:, 1]))

    # Get text size to create a background box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    text = str(room_id)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    border_color = (252, 252, 0)

    for i, corner in enumerate(preds):
        if i == len(preds) - 1:
            if closed:  # closing edge back to the first corner (polygons only)
                cv2.line(
                    room_map,
                    (round(corner[0]), round(corner[1])),
                    (round(preds[0][0]), round(preds[0][1])),
                    border_color,
                    2,
                )
        else:
            cv2.line(
                room_map,
                (round(corner[0]), round(corner[1])),
                (round(preds[i + 1][0]), round(preds[i + 1][1])),
                border_color,
                2,
            )
        cv2.circle(room_map, (round(corner[0]), round(corner[1])), 2, (0, 0, 255), 2)
        # cv2.putText(room_map, str(i), (round(corner[0]), round(corner[1])), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw white background box with transparency
        # overlay = room_map.copy()
        # cv2.addWeighted(overlay, 0.7, room_map, 0.3, 0, room_map)  # 70% opacity

        # Draw text
        if plot_text:
            cv2.rectangle(
                room_map,
                (centroid_x - text_width // 2 - 2, centroid_y - text_height // 2 - 2),
                (centroid_x + text_width // 2 + 2, centroid_y + text_height // 2 + 2),
                (255, 255, 255),  # (0, 0, 0),
                -1,
            )  # Filled rectangle
            cv2.putText(
                room_map,
                text,
                (centroid_x - text_width // 2, centroid_y + text_height // 2),
                font,
                font_scale,
                (0, 100, 0),
                thickness,
            )

    return room_map


def plot_density_map(sample, image_size, room_polys, pred_room_label_per_scene, plot_text=True):
    if not isinstance(sample, np.ndarray):
        density_map = np.transpose(sample.cpu().numpy(), [1, 2, 0])
    else:
        density_map = sample
    if density_map.shape[2] == 3:
        density_map = density_map * (image_size - 1)
    else:
        density_map = np.repeat(density_map, 3, axis=2) * (image_size - 1)
    pred_room_map = np.zeros([image_size, image_size, 3])

    for room_poly, room_id in zip(room_polys, pred_room_label_per_scene):
        pred_room_map = plot_room_map(room_poly, pred_room_map, room_id, im_size=image_size, plot_text=plot_text)

    alpha = 0.4  # Adjust for desired transparency
    pred_room_map = cv2.addWeighted(density_map.astype(np.uint8), alpha, pred_room_map.astype(np.uint8), 1 - alpha, 0)
    return pred_room_map


def plot_anno(img, annos, save_path, transformed=False, draw_poly=True, draw_bbx=True, thickness=2):
    """Visualize annotation"""
    img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
    num_inst = len(annos)

    bbx_color = (0, 255, 0)
    # poly_color = (0, 255, 0)
    for j in range(num_inst):
        if draw_bbx:
            bbox = annos[j]["bbox"]
            if transformed:
                start_point = (round(bbox[0]), round(bbox[1]))
                end_point = (round(bbox[2]), round(bbox[3]))
            else:
                start_point = (round(bbox[0]), round(bbox[1]))
                end_point = (round(bbox[0] + bbox[2]), round(bbox[1] + bbox[3]))
            # Blue color in BGR
            img = cv2.rectangle(img, start_point, end_point, bbx_color, thickness)

        if draw_poly:
            verts = annos[j]["segmentation"][0]
            if isinstance(verts, list):
                verts = np.array(verts)
            verts = verts.reshape(-1, 2)

            for i, corner in enumerate(verts):
                if i == len(verts) - 1:
                    cv2.line(
                        img,
                        (round(corner[0]), round(corner[1])),
                        (round(verts[0][0]), round(verts[0][1])),
                        (0, 252, 252),
                        1,
                    )
                else:
                    cv2.line(
                        img,
                        (round(corner[0]), round(corner[1])),
                        (round(verts[i + 1][0]), round(verts[i + 1][1])),
                        (0, 252, 252),
                        1,
                    )
                cv2.circle(img, (round(corner[0]), round(corner[1])), 2, (255, 0, 0), 2)
                cv2.putText(
                    img,
                    str(i),
                    (round(corner[0]), round(corner[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

    imsave(save_path, img)


def plot_coords(ax, ob, color=BLACK, zorder=1, alpha=1, linewidth=1):
    x, y = ob.xy
    ax.plot(x, y, color=color, zorder=zorder, alpha=alpha, linewidth=linewidth, solid_joinstyle="miter")


def plot_corners(ax, ob, color=BLACK, zorder=1, alpha=1):
    x, y = ob.xy
    ax.scatter(x, y, color=color, marker="o")


def get_angle(p1, p2):
    """Get the angle of this line with the horizontal axis."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return angle


def filled_arc(e1, e2, direction, radius, ax, color):
    """Draw arc for door"""
    angle = get_angle(e1, e2)
    if direction == "counterclock":
        theta1 = angle
        theta2 = angle + 90.0
    else:
        theta1 = angle - 90.0
        theta2 = angle
    circ = mpatches.Wedge(e1, radius, theta1, theta2, fill=True, color=color, linewidth=1, ec="#000000")
    ax.add_patch(circ)


def plot_semantic_rich_floorplan(polygons, file_name, prec=None, rec=None):
    """plot semantically-rich floorplan (i.e. with additional room label, door, window)"""

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    polygons_windows = []
    polygons_doors = []

    # Iterate over rooms to draw black outline
    for poly, poly_type in polygons:
        if len(poly) > 2:
            polygon = Polygon(poly)
            if poly_type != 16 and poly_type != 17:
                plot_coords(ax, polygon.exterior, alpha=1.0, linewidth=10)

    # Iterate over all predicted polygons (rooms, doors, windows)
    for poly, poly_type in polygons:
        if poly_type == "outqwall":  # unclear what is this?
            pass
        elif poly_type == 16:  # Door
            door_length = math.dist(poly[0], poly[1])
            polygons_doors.append([poly, poly_type, door_length])
        elif poly_type == 17:  # Window
            polygons_windows.append([poly, poly_type])
        else:  # regular room
            polygon = Polygon(poly)
            patch = PolygonPatch(polygon, facecolor="#FFFFFF", alpha=1.0, linewidth=0)
            ax.add_patch(patch)
            patch = PolygonPatch(
                polygon,
                facecolor=semantics_cmap[poly_type],
                alpha=0.5,
                linewidth=1,
                capstyle="round",
                edgecolor="#000000FF",
            )
            ax.add_patch(patch)
            ax.text(
                np.mean(poly[:, 0]),
                np.mean(poly[:, 1]),
                S3D_LABEL[poly_type],
                fontsize=6,
                horizontalalignment="center",
                verticalalignment="center",
                bbox=dict(facecolor="white", alpha=0.7),
            )

    # Compute door size statistics (median)
    door_median_size = np.median([door_length for (_, _, door_length) in polygons_doors])

    # Draw doors
    for poly, poly_type, door_size in polygons_doors:
        door_size_y = np.abs(poly[0, 1] - poly[1, 1])
        door_size_x = np.abs(poly[0, 0] - poly[1, 0])
        if door_size_y > door_size_x:
            if poly[1, 1] > poly[0, 1]:
                e1 = poly[0]
                e2 = poly[1]
            else:
                e1 = poly[1]
                e2 = poly[0]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, "clock", door_size, ax, "white")
            else:
                filled_arc(e1, e2, "clock", door_size / 2, ax, "white")
                filled_arc(e2, e1, "counterclock", door_size / 2, ax, "white")

        else:
            if poly[1, 0] > poly[0, 0]:
                e1 = poly[1]
                e2 = poly[0]
            else:
                e1 = poly[0]
                e2 = poly[1]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, "counterclock", door_size, ax, "white")
            else:
                filled_arc(e1, e2, "counterclock", door_size / 2, ax, "white")
                filled_arc(e2, e1, "clock", door_size / 2, ax, "white")

    # Draw windows
    for line, line_type in polygons_windows:
        line = LineString(line)
        poly = line.buffer(1.5, cap_style=2)
        if poly.is_empty:
            continue
        patch = PolygonPatch(poly, facecolor="#FFFFFF", alpha=1.0, linewidth=1, linestyle="dashed")
        ax.add_patch(patch)

    title = ""
    if prec is not None:
        title = "prec: " + str(round(prec * 100, 1)) + ", rec: " + str(round(rec * 100, 1))
    plt.title(file_name.split("/")[-1] + " " + title)
    plt.axis("equal")
    plt.axis("off")

    print(f">>> {file_name}")
    # fig.savefig(file_name[:-3]+'svg', dpi=fig.dpi, format='svg')
    fig.savefig(file_name, dpi=fig.dpi)


def plot_semantic_rich_floorplan_tight(
    polygons,
    file_name,
    prec=None,
    rec=None,
    plot_text=True,
    is_bw=False,
    door_window_index=[16, 17],
    img_w=256,
    img_h=256,
):
    """plot semantically-rich floorplan (i.e. with additional room label, door, window)"""

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    # Set figure size to exactly 256x256 pixels
    dpi = 100  # Standard screen DPI
    figsize = (img_w / dpi, img_h / dpi)  # Convert pixels to inches

    # Create square figure with fixed size
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Set equal aspect ratio and the limits to exactly match the coordinate space
    ax.set_aspect("equal")
    ax.set_xlim(0, img_w - 1)  # 255
    ax.set_ylim(0, img_h - 1)  # 255

    polygons_windows = []
    polygons_doors = []

    # Iterate over rooms to draw black outline
    for poly, poly_type in polygons:
        if len(poly) > 2:
            polygon = Polygon(poly)

            if poly_type not in door_window_index:
                plot_coords(ax, polygon.exterior, alpha=1.0, linewidth=10)

    # Iterate over all predicted polygons (rooms, doors, windows)
    for poly, poly_type in polygons:
        if poly_type == "outqwall":  # unclear what is this?
            pass
        elif poly_type == door_window_index[0]:  # Door
            door_length = math.dist(poly[0], poly[1])
            polygons_doors.append([poly, poly_type, door_length])
        elif poly_type == door_window_index[1]:  # Window
            polygons_windows.append([poly, poly_type])
        else:  # regular room
            if len(poly) < 3:
                continue
            polygon = Polygon(poly)
            patch = PolygonPatch(polygon, facecolor="#FFFFFF", alpha=1.0, linewidth=0)
            ax.add_patch(patch)
            if not is_bw:
                patch = PolygonPatch(
                    polygon,
                    facecolor=semantics_cmap[poly_type],
                    alpha=0.5,
                    linewidth=1,
                    capstyle="round",
                    edgecolor="#000000FF",
                )
                ax.add_patch(patch)
            if plot_text:
                ax.text(
                    np.mean(poly[:, 0]),
                    np.mean(poly[:, 1]),
                    S3D_LABEL[poly_type],
                    size=6,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

    # Compute door size statistics (median)
    door_median_size = np.median([door_length for (_, _, door_length) in polygons_doors])

    # Draw doors
    for poly, poly_type, door_size in polygons_doors:
        door_size_y = np.abs(poly[0, 1] - poly[1, 1])
        door_size_x = np.abs(poly[0, 0] - poly[1, 0])
        if door_size_y > door_size_x:
            if poly[1, 1] > poly[0, 1]:
                e1 = poly[0]
                e2 = poly[1]
            else:
                e1 = poly[1]
                e2 = poly[0]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, "clock", door_size, ax, "white")
            else:
                filled_arc(e1, e2, "clock", door_size / 2, ax, "white")
                filled_arc(e2, e1, "counterclock", door_size / 2, ax, "white")

        else:
            if poly[1, 0] > poly[0, 0]:
                e1 = poly[1]
                e2 = poly[0]
            else:
                e1 = poly[0]
                e2 = poly[1]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, "counterclock", door_size, ax, "white")
            else:
                filled_arc(e1, e2, "counterclock", door_size / 2, ax, "white")
                filled_arc(e2, e1, "clock", door_size / 2, ax, "white")

    # Draw windows
    for line, line_type in polygons_windows:
        line = LineString(line)
        poly = line.buffer(1.5, cap_style=2)
        if poly.is_empty:
            continue
        patch = PolygonPatch(poly, facecolor="#FFFFFF", alpha=1.0, linewidth=1, linestyle="dashed")
        ax.add_patch(patch)

    if plot_text:
        title = ""
        if prec is not None:
            title = "prec: " + str(round(prec * 100, 1)) + ", rec: " + str(round(rec * 100, 1))
        plt.title(file_name.split("/")[-1] + " " + title)

    # plt.axis('equal')
    plt.axis("off")

    print(f">>> {file_name}")
    # fig.savefig(file_name[:-3]+'svg', dpi=fig.dpi, format='svg')
    if is_bw:
        plt.set_cmap(get_cmap("gray"))

    fig.savefig(file_name, dpi=dpi, bbox_inches="tight", pad_inches=0)


def plot_semantic_rich_floorplan_nicely(
    polygons,
    file_name,
    prec=None,
    rec=None,
    plot_text=True,
    is_bw=False,
    door_window_index=[16, 17],
    img_w=256,
    img_h=256,
    semantics_label_mapping=S3D_LABEL,
):
    """plot semantically-rich floorplan (i.e. with additional room label, door, window)"""

    # Set figure size to exactly 256x256 pixels
    dpi = 150  # Standard screen DPI
    figsize = (img_w / dpi, img_h / dpi)  # Convert pixels to inches

    # Create square figure with fixed size
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    # Set equal aspect ratio and the limits to exactly match the coordinate space
    # ax.set_aspect('equal')
    # ax.set_xlim(0, img_w - 1)
    # ax.set_ylim(0, img_h - 1)

    # Disable autoscaling
    ax.autoscale(False)

    # Disable adjusting automatically
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    polygons_windows = []
    polygons_doors = []

    # Iterate over rooms to draw black outline
    for poly, poly_type in polygons:
        if len(poly) > 2:
            polygon = Polygon(poly)

            if poly_type not in door_window_index:
                plot_coords(ax, polygon.exterior, alpha=1.0, linewidth=2)

    # Iterate over all predicted polygons (rooms, doors, windows)
    for poly, poly_type in polygons:
        if poly_type == door_window_index[0]:  # Door
            door_length = math.dist(poly[0], poly[1])
            polygons_doors.append([poly, poly_type, door_length])
        elif poly_type == door_window_index[1]:  # Window
            polygons_windows.append([poly, poly_type])
        else:  # regular room
            polygon = Polygon(poly)
            patch = PolygonPatch(polygon, facecolor="#FFFFFF", alpha=1.0, linewidth=0)
            ax.add_patch(patch)
            if not is_bw:
                patch = PolygonPatch(
                    polygon,
                    facecolor=semantics_cmap[poly_type],
                    alpha=0.5,
                    linewidth=1,
                    capstyle="round",
                    edgecolor="#000000FF",
                )
                ax.add_patch(patch)
            if plot_text:
                ax.text(
                    np.mean(poly[:, 0]),
                    np.mean(poly[:, 1]),
                    semantics_label_mapping[poly_type],
                    fontsize=6,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.7),
                )

    # Compute door size statistics (median)
    door_median_size = np.median([door_length for (_, _, door_length) in polygons_doors])

    # Draw doors
    for poly, poly_type, door_size in polygons_doors:
        door_size_y = np.abs(poly[0, 1] - poly[1, 1])
        door_size_x = np.abs(poly[0, 0] - poly[1, 0])
        if door_size_y > door_size_x:
            if poly[1, 1] > poly[0, 1]:
                e1 = poly[0]
                e2 = poly[1]
            else:
                e1 = poly[1]
                e2 = poly[0]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, "clock", door_size, ax, "white")
            else:
                filled_arc(e1, e2, "clock", door_size / 2, ax, "white")
                filled_arc(e2, e1, "counterclock", door_size / 2, ax, "white")

        else:
            if poly[1, 0] > poly[0, 0]:
                e1 = poly[1]
                e2 = poly[0]
            else:
                e1 = poly[0]
                e2 = poly[1]

            if door_size < door_median_size * 1.5:
                filled_arc(e1, e2, "counterclock", door_size, ax, "white")
            else:
                filled_arc(e1, e2, "counterclock", door_size / 2, ax, "white")
                filled_arc(e2, e1, "clock", door_size / 2, ax, "white")

    # Draw windows
    for line, line_type in polygons_windows:
        line = LineString(line)
        poly = line.buffer(1.5, cap_style=2)
        if poly.is_empty:
            continue
        patch = PolygonPatch(poly, facecolor="#FFFFFF", alpha=1.0, linewidth=1, linestyle="dashed")
        ax.add_patch(patch)

    if plot_text:
        title = ""
        if prec is not None:
            title = "prec: " + str(round(prec * 100, 1)) + ", rec: " + str(round(rec * 100, 1))
        plt.title(file_name.split("/")[-1] + " " + title)

    print(f">>> {file_name}")
    plt.axis("equal")
    plt.axis("off")
    # fig.savefig(file_name[:-3]+'svg', dpi=fig.dpi, format='svg')
    if is_bw:
        plt.set_cmap(get_cmap("gray"))
    fig.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_semantic_rich_floorplan_opencv(
    polygons,
    file_name,
    img_w=256,
    img_h=256,
    door_window_index=[16, 17],
    semantics_label_mapping=S3D_LABEL,
    is_bw=False,
    plot_text=True,
    one_color=False,
    scale=1,
    is_sem=True,
    highlight_indices=None,
):
    """
    Plot semantically-rich floorplan using OpenCV with improved quality.

    Args:
        polygons (list): A list of [polygon, type] or [polygon, type, is_line] entries, where polygon is a
            list of (x, y) coordinates; is_line=True draws the shape as an open line (not filled or closed).
        file_name (str): Path to save the output image.
        img_w (int): Width of the output image.
        img_h (int): Height of the output image.
        door_window_index (list): Indices for door and window types.
        semantics_label_mapping (dict): Mapping from polygon type to semantic label.
        is_bw (bool): If True, use black and white colors only.
        line_thickness (int): Thickness of lines for polygons and doors/windows.
        text_padding (int): Padding around text labels.
        font_scale (float): Scale factor for text size.
        room_alpha (float): Transparency for room colors (0.0-1.0).
        anti_aliasing (bool): Whether to use anti-aliasing for lines.
    """
    line_thickness = 2
    text_padding = 1
    font_scale = 0.25
    room_alpha = 0.6

    if scale != 1:
        new_polygons = []
        for poly, *rest in polygons:
            poly = (poly * scale).round().astype(np.int32)
            new_polygons.append([poly, *rest])
        polygons = new_polygons

    if one_color:
        colors = ["#FFD700"]
    else:
        colors = list(qualitative.Set3) + list(qualitative.Dark2)
        rgb_string_to_tuple = lambda rgb_string: tuple(float(x) / 255 for x in rgb_string.strip("rgb()").split(","))
        colors = [to_hex(rgb_string_to_tuple(x)) for x in colors]

    # Create a white background image (more conventional for floorplans)
    if is_bw:
        image = np.ones((img_h, img_w), dtype=np.uint8) * 255  # White grayscale image
    else:
        image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255  # White RGB image

    # Create a separate layer for room colors
    overlay = image.copy()

    highlight_indices_set = set(highlight_indices) if highlight_indices is not None else None

    # Track polygons for each type for proper layering
    room_polygons = []
    door_polygons = []
    window_polygons = []
    line_polygons = []  # open lines (entries flagged is_line)

    # Sort polygons by type (track original index for highlight support)
    for poly_idx, (poly, poly_type, *flags) in enumerate(polygons):
        if len(poly) < 2:  # Skip invalid polygons
            continue

        points = np.array(poly, dtype=np.int32)

        if flags and flags[0]:  # Open line: drawn as a stroke, never filled or closed
            line_polygons.append((points, poly_type, poly_idx))
        elif door_window_index and poly_type == door_window_index[0]:  # Door
            door_polygons.append((points, poly_type, poly_idx))
        elif door_window_index and poly_type == door_window_index[1]:  # Window
            window_polygons.append((points, poly_type, poly_idx))
        else:  # Room
            room_polygons.append((points, poly_type, poly_idx))

    # Draw rooms first (bottom layer)
    for room_id, (points, poly_type, orig_idx) in enumerate(room_polygons):
        # Fill room with color
        if not is_bw:
            # If highlight_indices is set, use gold for non-highlighted polygons
            if highlight_indices_set is not None and orig_idx not in highlight_indices_set:
                rgb_color = ImageColor.getcolor("#FFD700", "RGB")
            elif not is_sem:
                rgb_color = ImageColor.getcolor(colors[room_id % len(colors)], "RGB")
            else:
                rgb_color = ImageColor.getcolor(colors[poly_type % len(colors)], "RGB")

            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
            cv2.fillPoly(overlay, [points], color=bgr_color)
        else:
            # Use light gray for rooms in BW mode
            cv2.fillPoly(overlay, [points], color=(240, 240, 240))

        # Draw room outline
        line_type = cv2.LINE_AA
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 0), thickness=line_thickness, lineType=line_type)

    # Blend overlay with transparency
    cv2.addWeighted(overlay, room_alpha, image, 1 - room_alpha, 0, image)

    # Draw open lines, coloured by type like rooms
    for points, poly_type, _orig_idx in line_polygons:
        if is_bw:
            line_color = (0, 0, 0)
        else:
            rgb_color = ImageColor.getcolor(colors[poly_type % len(colors)], "RGB")
            line_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        cv2.polylines(image, [points], isClosed=False, color=line_color, thickness=line_thickness, lineType=cv2.LINE_AA)

    # Draw doors with proper styling
    for points, _, _orig_idx in door_polygons:
        if len(points) >= 2:
            # For doors, we can improve by drawing arcs to represent swing
            # Here we draw them as thick lines with distinctive color
            door_color = (100, 100, 100) if is_bw else (0, 0, 255)  # Gray for BW, Red for RGB
            line_type = cv2.LINE_AA
            cv2.polylines(
                image, [points], isClosed=False, color=door_color, thickness=line_thickness * 2, lineType=line_type
            )

    # Draw windows with dashed styling
    for points, _, _orig_idx in window_polygons:
        if len(points) >= 2:
            window_color = (150, 150, 150) if is_bw else (255, 0, 0)  # Gray for BW, Blue for RGB

            # Create dashed line effect for windows
            if len(points) == 2:
                # For a simple line window
                pt1, pt2 = points[0], points[1]
                dash_length = 5

                # Calculate line parameters
                length = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
                if length > 0:
                    num_dashes = max(2, int(length / (2 * dash_length)))

                    for i in range(num_dashes):
                        start_ratio = i / num_dashes
                        end_ratio = (i + 0.5) / num_dashes

                        start_x = int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio)
                        start_y = int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
                        end_x = int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio)
                        end_y = int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)

                        line_type = cv2.LINE_AA
                        cv2.line(
                            image,
                            (start_x, start_y),
                            (end_x, end_y),
                            window_color,
                            thickness=line_thickness,
                            lineType=line_type,
                        )
            else:
                # For multi-point windows
                line_type = cv2.LINE_AA
                cv2.polylines(
                    image, [points], isClosed=True, color=window_color, thickness=line_thickness, lineType=line_type
                )

    if plot_text:
        # Add room labels
        for i, (points, poly_type, _orig_idx) in enumerate(room_polygons):
            # if i > 1: continue # TODO:test
            # Calculate the centroid for text placement
            M = cv2.moments(points)
            if M["m00"] != 0:  # Avoid division by zero
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])

                # Get room label
                label = semantics_label_mapping[poly_type]

                # Get text size for centering and background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

                # Calculate text background rectangle
                text_x = centroid_x - text_size[0] // 2
                text_y = centroid_y + text_size[1] // 2

                # Create background for text
                rect_top_left = (text_x - text_padding, text_y - text_size[1] - text_padding)
                rect_bottom_right = (text_x + text_size[0] + text_padding, text_y + text_padding)

                # Draw semi-transparent white background for text
                background = image.copy()
                cv2.rectangle(background, rect_top_left, rect_bottom_right, (255, 255, 255), -1)

                # Blend the background
                cv2.addWeighted(background, 0.7, image, 0.3, 0, image)

                # Draw the text
                cv2.putText(
                    image,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),  # Black text
                    1,  # Thickness
                    cv2.LINE_AA,  # Anti-aliased text
                )

    # Add border around the image for better framing
    # cv2.rectangle(image, (0, 0), (img_w-1, img_h-1), (0, 0, 0), 1, cv2.LINE_AA)

    # Save with high quality
    if file_name is not None:
        if is_bw:
            cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Saved improved floorplan to {file_name}")

    return image


def draw_dashed_line(image, pt1, pt2, color, thickness, dash_length=10):
    """Draw a dashed line between two points."""
    # Calculate the Euclidean distance between the points
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    # Calculate the number of dashes
    num_dashes = int(dist // dash_length)
    # Calculate the direction vector
    direction = (np.array(pt2) - np.array(pt1)) / dist
    for i in range(num_dashes):
        start = pt1 + direction * (i * dash_length)
        end = pt1 + direction * ((i + 0.5) * dash_length)
        cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)


def draw_dashed_polyline(image, points, color, thickness, dash_length=10, gap_length=5):
    """
    Draws a dashed polyline with evenly spaced dashes along the entire path.

    Parameters:
    - image: The image on which to draw.
    - points: List of points defining the polyline.
    - color: Color of the dashes (BGR tuple).
    - thickness: Thickness of the dashes.
    - dash_length: Length of each dash.
    - gap_length: Length of the gap between dashes.
    """
    if len(points) < 2:
        return

    # Convert points to numpy array for vectorized operations
    pts = np.array(points, dtype=np.float32)

    # Calculate the total length of the polyline
    segment_lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total_length = np.sum(segment_lengths)

    # Determine number of dashes
    pattern_length = dash_length + gap_length
    num_dashes = int(total_length // pattern_length)

    # Generate dash start positions along the total length
    dash_positions = np.arange(0, num_dashes * pattern_length, pattern_length)

    # Initialize variables to track the current segment
    seg_idx = 0
    seg_start = pts[0]
    seg_end = pts[1]
    seg_length = segment_lengths[0]
    seg_vector = (seg_end - seg_start) / seg_length
    seg_pos = 0.0  # Position along the current segment

    for pos in dash_positions:
        # Advance to the segment containing the current dash
        while seg_pos + seg_length < pos:
            seg_pos += seg_length
            seg_idx += 1
            if seg_idx >= len(pts) - 1:
                return
            seg_start = pts[seg_idx]
            seg_end = pts[seg_idx + 1]
            seg_length = segment_lengths[seg_idx]
            seg_vector = (seg_end - seg_start) / seg_length

        # Calculate start and end points of the dash
        offset = pos - seg_pos
        start_point = seg_start + seg_vector * offset
        end_offset = min(dash_length, seg_length - offset)
        end_point = start_point + seg_vector * end_offset

        # Draw the dash
        cv2.line(
            image, tuple(np.round(start_point).astype(int)), tuple(np.round(end_point).astype(int)), color, thickness
        )


def plot_semantic_rich_floorplan_opencv_figure(
    polygons,
    file_name,
    img_w=256,
    img_h=256,
    door_window_index=[16, 17],
    semantics_label_mapping=S3D_LABEL,
    is_bw=False,
    plot_text=True,
    one_color=False,
):
    """
    Plot semantically-rich floorplan using OpenCV with improved quality.

    Args:
        polygons (list): A list of polygons, where each polygon is a list of (x, y) coordinates.
        file_name (str): Path to save the output image.
        img_w (int): Width of the output image.
        img_h (int): Height of the output image.
        door_window_index (list): Indices for door and window types.
        semantics_label_mapping (dict): Mapping from polygon type to semantic label.
        is_bw (bool): If True, use black and white colors only.
        line_thickness (int): Thickness of lines for polygons and doors/windows.
        text_padding (int): Padding around text labels.
        font_scale (float): Scale factor for text size.
        room_alpha (float): Transparency for room colors (0.0-1.0).
        anti_aliasing (bool): Whether to use anti-aliasing for lines.
    """
    line_thickness = 2
    text_padding = 1
    font_scale = 1.0
    room_alpha = 0.6

    if img_w != 256:
        new_polygons = []
        for poly, poly_label in polygons:
            poly = (poly * img_w / 256).round().astype(np.int32)
            new_polygons.append([poly, poly_label])
        polygons = new_polygons

    if one_color:
        colors = ["#FFD700"]
    else:
        # colors = [to_hex(x) for x in qualitative.Light24]
        # TODO
        colors = ["#FFFFFF"] * len(qualitative.Light24)
        colors[polygons[0][1]] = "#FF9616"  # red
        colors[polygons[1][1]] = "#FE00CE"  # green

    # cmap = get_cmap('tab20', 20)
    # colors = [to_hex(cmap(x)) for x in np.linspace(0, 1, 20)]  # Convert to hex
    # Create a white background image (more conventional for floorplans)
    if is_bw:
        image = np.ones((img_h, img_w), dtype=np.uint8) * 255  # White grayscale image
    else:
        image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255  # White RGB image

    # Create a separate layer for room colors
    overlay = image.copy()

    # Track polygons for each type for proper layering
    room_polygons = []
    door_polygons = []
    window_polygons = []

    # Sort polygons by type
    for poly, poly_type in polygons:
        if len(poly) < 2:  # Skip invalid polygons
            continue

        points = np.array(poly, dtype=np.int32)

        if poly_type == door_window_index[0]:  # Door
            door_polygons.append((points, poly_type))
        elif poly_type == door_window_index[1]:  # Window
            window_polygons.append((points, poly_type))
        else:  # Room
            room_polygons.append((points, poly_type))

    # Draw rooms first (bottom layer)
    for room_id, (points, poly_type) in enumerate(room_polygons):
        # TODO:test
        if room_id > 1:
            poly_type = room_polygons[0][1] + 1

        # Fill room with color
        if not is_bw:
            # Get RGB color from semantics_cmap and convert from RGB to BGR for OpenCV
            # if not plot_text:
            #     rgb_color = ImageColor.getcolor(colors[room_id % len(colors)], "RGB")
            # else:
            #     rgb_color = ImageColor.getcolor(colors[poly_type % len(colors)], "RGB")
            # TODO
            rgb_color = ImageColor.getcolor(colors[poly_type % len(colors)], "RGB")
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

            cv2.fillPoly(overlay, [points], color=bgr_color)
        else:
            # Use light gray for rooms in BW mode
            cv2.fillPoly(overlay, [points], color=(240, 240, 240))

        # # Draw room outline
        # if room_id > 1:
        #     # Draw dashed room outline
        #     for i in range(len(points)):
        #         pt1 = points[i]
        #         pt2 = points[(i + 1) % len(points)]  # Wrap around to the first point
        #         # draw_dashed_line(image, pt1, pt2, color=(0, 0, 0), thickness=line_thickness, dash_length=10)
        #         draw_dashed_polyline(image, points, color=(0, 0, 0), thickness=line_thickness, dash_length=5, gap_length=5)
        # else:
        line_type = cv2.LINE_AA
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 0), thickness=line_thickness, lineType=line_type)

    # Blend overlay with transparency
    cv2.addWeighted(overlay, room_alpha, image, 1 - room_alpha, 0, image)

    # Draw doors with proper styling
    for points, _ in door_polygons:
        if len(points) >= 2:
            # For doors, we can improve by drawing arcs to represent swing
            # Here we draw them as thick lines with distinctive color
            door_color = (100, 100, 100) if is_bw else (0, 0, 255)  # Gray for BW, Red for RGB
            line_type = cv2.LINE_AA
            cv2.polylines(
                image, [points], isClosed=False, color=door_color, thickness=line_thickness * 2, lineType=line_type
            )

    # Draw windows with dashed styling
    for points, _ in window_polygons:
        if len(points) >= 2:
            window_color = (150, 150, 150) if is_bw else (255, 0, 0)  # Gray for BW, Blue for RGB

            # Create dashed line effect for windows
            if len(points) == 2:
                # For a simple line window
                pt1, pt2 = points[0], points[1]
                dash_length = 5

                # Calculate line parameters
                length = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
                if length > 0:
                    num_dashes = max(2, int(length / (2 * dash_length)))

                    for i in range(num_dashes):
                        start_ratio = i / num_dashes
                        end_ratio = (i + 0.5) / num_dashes

                        start_x = int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio)
                        start_y = int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
                        end_x = int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio)
                        end_y = int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)

                        line_type = cv2.LINE_AA
                        cv2.line(
                            image,
                            (start_x, start_y),
                            (end_x, end_y),
                            window_color,
                            thickness=line_thickness,
                            lineType=line_type,
                        )
            else:
                # For multi-point windows
                line_type = cv2.LINE_AA
                cv2.polylines(
                    image, [points], isClosed=True, color=window_color, thickness=line_thickness, lineType=line_type
                )

    if plot_text:
        # Add room labels
        for i, (points, poly_type) in enumerate(room_polygons):
            if i > 1:
                continue  # TODO:test
            # Calculate the centroid for text placement
            M = cv2.moments(points)
            if M["m00"] != 0:  # Avoid division by zero
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])

                # Get room label
                label = semantics_label_mapping[poly_type]

                # Get text size for centering and background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]

                # Calculate text background rectangle
                text_x = centroid_x - text_size[0] // 2
                text_y = centroid_y + text_size[1] // 2

                # Create background for text
                rect_top_left = (text_x - text_padding, text_y - text_size[1] - text_padding)
                rect_bottom_right = (text_x + text_size[0] + text_padding, text_y + text_padding)

                # Draw semi-transparent white background for text
                background = image.copy()
                cv2.rectangle(background, rect_top_left, rect_bottom_right, (255, 255, 255), -1)

                # Blend the background
                cv2.addWeighted(background, 0.7, image, 0.3, 0, image)

                # Draw the text
                cv2.putText(
                    image,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),  # Black text
                    1,  # Thickness
                    cv2.LINE_AA,  # Anti-aliased text
                )

    # Add border around the image for better framing
    # cv2.rectangle(image, (0, 0), (img_w-1, img_h-1), (0, 0, 0), 1, cv2.LINE_AA)

    # Save with high quality
    if is_bw:
        cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"Saved improved floorplan to {file_name}")

    return image  # Return the image for optional further processing or visualization


def sort_polygons_by_matching(matching_pred2gt, pred_polygons, gt_polygons):
    """
    Sorts pred_polygons and gt_polygons based on the matching indices.

    Args:
        matching_pred2gt (list): List of matching indices from pred to gt.
        pred_polygons (list): List of predicted polygons.
        gt_polygons (list): List of ground truth polygons.

    Returns:
        tuple: (sorted_pred_polygons, sorted_gt_polygons)
    """
    sorted_pred_polygons = []  # Keep the order of pred_polygons as is
    sorted_gt_polygons = []
    pred_mask = []
    gt_mask = []
    remaining_pred_polygons = []

    for i, match_idx in enumerate(matching_pred2gt):
        if match_idx == -1:
            # sorted_gt_polygons.append(None)  # No match, insert placeholder
            remaining_pred_polygons.append(pred_polygons[i])
            continue
        else:
            sorted_pred_polygons.append(pred_polygons[i])
            sorted_gt_polygons.append(gt_polygons[match_idx])
            gt_mask.append(1)
            pred_mask.append(1)

    sorted_pred_polygons.extend(remaining_pred_polygons)
    pred_mask.extend([0] * len(remaining_pred_polygons))

    for i in range(len(gt_polygons)):
        if i not in matching_pred2gt:
            sorted_gt_polygons.append(gt_polygons[i])
            gt_mask.append(0)

    return sorted_pred_polygons, sorted_gt_polygons, pred_mask, gt_mask


def concat_floorplan_maps(gt_floorplan_map, floorplan_map, plot_statistics={}):
    pad_color = (0, 0, 0) if gt_floorplan_map.shape[2] == 3 else (0, 0, 0, 0)
    padding = np.full((gt_floorplan_map.shape[0], 10, gt_floorplan_map.shape[2]), pad_color, dtype=np.uint8)
    # Concatenate pred_room_map, padding, and gt_room_map
    concatenated_map = cv2.hconcat([gt_floorplan_map, padding, floorplan_map])
    top_padding = np.full((100, concatenated_map.shape[1], concatenated_map.shape[2]), pad_color, dtype=np.uint8)

    # Add text for f1 and missing_rate
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255) if gt_floorplan_map.shape[2] == 3 else (0, 0, 255, 255)  # White text
    thickness = 2
    line_type = cv2.LINE_AA

    # Position for the text
    text_f1 = (
        f"F1: {plot_statistics['f1']:.2f}, Prec: {plot_statistics['prec']:.2f}, Rec: {plot_statistics['rec']:.2f}"
    )
    text_missing_rate = f"Missing Rate: {plot_statistics['missing_rate']:.2f}, {plot_statistics['num_preds']}/{plot_statistics['num_matched_preds']}/{plot_statistics['num_gt']}"
    text_position_f1 = (10, 30)  # Position within the top padding
    text_position_missing_rate = (10, 70)  # Adjusted position for the second line

    # Overlay text on the top padding
    cv2.putText(top_padding, text_f1, text_position_f1, font, font_scale, font_color, thickness, line_type)
    cv2.putText(
        top_padding, text_missing_rate, text_position_missing_rate, font, font_scale, font_color, thickness, line_type
    )

    # Concatenate the top padding with the concatenated_map
    final_map = cv2.vconcat([top_padding, concatenated_map])
    return final_map
