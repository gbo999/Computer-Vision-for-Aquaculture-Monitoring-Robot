"""Compare predicted bounding boxes from YOLOv8 keypoint outputs (runs/pose/predict83/labels)
with ImageJ-measured bounding boxes stored in
fifty_one/measurements/data/length_analysis_new_split_shai_exuviae.csv.

For each image we:
1. Load the YOLO label file corresponding to that image.
2. Select the prediction with the largest bounding-box area (in pixels).
3. Convert its normalized (x_center, y_center, width, height) values to
   absolute pixel coordinates using the resolution of the colourized image
   produced by the YOLO pipeline.
4. Store the resulting BX_pred, BY_pred, Width_pred, Height_pred values and
   compute their absolute differences from the ImageJ BX, BY, Width, Height.
5. Save a CSV alongside the original measurements called
   length_analysis_new_split_shai_exuviae_with_yolo.csv

Run this file from the project root:
   python scripts/bbox_compare_shai_exuviae.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image

# --- Configuration ------------------------------------------------------------------
LABELS_DIR = Path("training and val output/runs/pose/predict83/labels")
IMAGES_DIR = LABELS_DIR.parent  # one level up (runs/pose/predict83)
CSV_PATH = Path("fifty_one and analysis/measurements/exuviae/spreadsheet_files/length_analysis_new_split_shai_exuviae.csv")
OUTPUT_CSV_PATH = CSV_PATH.with_name(
    CSV_PATH.stem + "_with_yolo" + CSV_PATH.suffix
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_label_file(path: Path) -> List[Tuple[float, float, float, float, List[Tuple[float, float, float]]]]:
    """Return list of YOLO predictions found in *path*.

    Each prediction is represented as a tuple:
        (xc, yc, w, h, keypoints)

    where:
        - xc, yc, w, h are *normalized* bounding-box parameters in the range [0,1].
        - *keypoints* is a ``List`` of ``(x, y, v)`` triplets, also normalized.

    The function is aware of the YOLOv8 *keypoint* label format::

        class xc yc w h kp1x kp1y kp1v kp2x kp2y kp2v ...

    and is therefore able to keep all the keypoint information we will need later
    on for extracting the *rostrum* and *tail* locations.
    """

    preds: List[Tuple[float, float, float, float, List[Tuple[float, float, float]]]] = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                # Not enough information – skip this line
                continue

            try:
                # The first element is the class id; not used here
                xc, yc, w, h = map(float, parts[1:5])

                # Parse keypoints (if present) – they come in (x, y, v) triplets
                kp_values = list(map(float, parts[5:]))
                keypoints: List[Tuple[float, float, float]] = []
                for i in range(0, len(kp_values), 3):
                    if i + 2 >= len(kp_values):
                        break  # incomplete triplet – ignore
                    keypoints.append(
                        (
                            kp_values[i],       # x
                            kp_values[i + 1],   # y
                            kp_values[i + 2],   # v (visibility/confidence)
                        )
                    )

                preds.append((xc, yc, w, h, keypoints))

            except ValueError:
                # Any failure in parsing numbers – skip the problematic line
                continue

    return preds


def bbox_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two pixel-space boxes given as (x, y, w, h)."""

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    # Convert to (x1, y1, x2, y2)
    a_x1, a_y1, a_x2, a_y2 = ax, ay, ax + aw, ay + ah
    b_x1, b_y1, b_x2, b_y2 = bx, by, bx + bw, by + bh

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh

    return inter_area / (area_a + area_b - inter_area)


def choose_best_match(
    preds: List[Tuple[float, float, float, float, List[Tuple[float, float, float]]]],
    gt_box_px: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float, List[Tuple[float, float, float]]]:
    """Return the prediction whose bounding box has the highest IoU with *gt_box_px*."""

    if not preds:
        raise ValueError("No predictions provided")

    best_pred = None
    best_iou = -1.0

    for pred in preds:
        xc, yc, w_n, h_n, kps = pred
        px_box = normalized_to_pixel(xc, yc, w_n, h_n, img_w, img_h)
        iou = bbox_iou(px_box, gt_box_px)
        if iou > best_iou:
            best_iou = iou
            best_pred = pred

    return best_pred if best_pred is not None else preds[0]


def normalized_to_pixel(
    xc: float,
    yc: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert normalized bbox to pixel (BX, BY, Width, Height).

    BX and BY correspond to the top-left corner.
    """
    px_width = w * img_w
    px_height = h * img_h
    px_x = (xc - w / 2) * img_w
    px_y = (yc - h / 2) * img_h
    return px_x, px_y, px_width, px_height


# --- Main processing ----------------------------------------------------------------

def main() -> None:
    logging.info("Loading ground-truth measurements from %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Prepare columns for YOLO predictions and differences
    for col in [
        "BX_pred",
        "BY_pred",
        "Width_pred",
        "Height_pred",
        "rostrum_x_pred",
        "rostrum_y_pred",
        "tail_x_pred",
        "tail_y_pred",
        "rostrum_x_norm",
        "rostrum_y_norm",
        "tail_x_norm",
        "tail_y_norm",
        "rostrum_x_norm_ref",
        "rostrum_y_norm_ref",
        "tail_x_norm_ref",
        "tail_y_norm_ref",
        "pixels_total_length_pred",
        "pixels_total_length_err",
    ]:
        df[col] = math.nan
    for col in ["BX_err", "BY_err", "Width_err", "Height_err"]:
        df[col] = math.nan

    for idx, row in df.iterrows():
        base_name = row["image_name"]  # e.g. undistorted_GX010191_8_309
        label_file = LABELS_DIR / f"colored_{base_name}.txt"
        image_file_jpg = IMAGES_DIR / f"colored_{base_name}.jpg"

        if not label_file.exists():
            logging.warning("Label file not found: %s", label_file)
            continue
        if not image_file_jpg.exists():
            logging.warning("Image file not found: %s", image_file_jpg)
            continue

        # Parse YOLO predictions (bounding boxes + keypoints)
        preds = parse_label_file(label_file)

        if not preds:
            logging.warning("No predictions in %s", label_file)
            continue

        # Image dimensions (needed for IoU computation)
        with Image.open(image_file_jpg) as img:
            img_w, img_h = img.size

        # Ground-truth data
        gt_box_px = (row["BX"], row["BY"], row["Width"], row["Height"])
        gt_pixel_length = row["pixels_total_length"]

        # Select prediction that best matches tail→rostrum length
        (
            best_pred,
            pixel_length_pred,
            pixel_length_err,
        ) = choose_best_match_by_length(preds, gt_pixel_length, img_w, img_h)

        xc, yc, w_norm, h_norm, keypoints = best_pred

        bx_pred, by_pred, w_pred, h_pred = normalized_to_pixel(
            xc, yc, w_norm, h_norm, img_w, img_h
        )

        # Store bounding-box predictions
        df.at[idx, "BX_pred"] = bx_pred
        df.at[idx, "BY_pred"] = by_pred
        df.at[idx, "Width_pred"] = w_pred
        df.at[idx, "Height_pred"] = h_pred

        # Keypoints – extract *rostrum* (index 2) and *tail* (index 3)
        try:
            rostrum_kp = keypoints[2]  # (x, y, v)
            tail_kp = keypoints[3]

            rostrum_x_px = rostrum_kp[0] * img_w
            rostrum_y_px = rostrum_kp[1] * img_h
            tail_x_px = tail_kp[0] * img_w
            tail_y_px = tail_kp[1] * img_h

            # Normalized keypoint values (already in [0,1])
            df.at[idx, "rostrum_x_norm"] = rostrum_kp[0]
            df.at[idx, "rostrum_y_norm"] = rostrum_kp[1]
            df.at[idx, "tail_x_norm"] = tail_kp[0]
            df.at[idx, "tail_y_norm"] = tail_kp[1]

            # Pixel-space coordinates
            df.at[idx, "rostrum_x_pred"] = rostrum_x_px
            df.at[idx, "rostrum_y_pred"] = rostrum_y_px
            df.at[idx, "tail_x_pred"] = tail_x_px
            df.at[idx, "tail_y_pred"] = tail_y_px

            # Coordinates normalised to reference resolution (5312 x 2988)
            scale_factor_w = 5312.0 / img_w
            scale_factor_h = 2988.0 / img_h
            df.at[idx, "rostrum_x_norm_ref"] = (rostrum_x_px * scale_factor_w) / 5312.0
            df.at[idx, "rostrum_y_norm_ref"] = (rostrum_y_px * scale_factor_h) / 2988.0
            df.at[idx, "tail_x_norm_ref"] = (tail_x_px * scale_factor_w) / 5312.0
            df.at[idx, "tail_y_norm_ref"] = (tail_y_px * scale_factor_h) / 2988.0

            # Pixel-length comparison columns
            df.at[idx, "pixels_total_length_pred"] = pixel_length_pred
            df.at[idx, "pixels_total_length_err"] = pixel_length_err

        except IndexError:
            # Not enough keypoints – keep NaN and warn the user
            logging.warning(
                "Prediction in %s does not contain the expected number of keypoints (got %d)",
                label_file,
                len(keypoints),
            )

        # Differences (absolute)
        df.at[idx, "BX_err"] = abs(row["BX"] - bx_pred)
        df.at[idx, "BY_err"] = abs(row["BY"] - by_pred)
        df.at[idx, "Width_err"] = abs(row["Width"] - w_pred)
        df.at[idx, "Height_err"] = abs(row["Height"] - h_pred)

    logging.info("Saving combined results to %s", OUTPUT_CSV_PATH)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    # Quick summary statistics
    err_cols = ["BX_err", "BY_err", "Width_err", "Height_err"]
    summary = df[err_cols].describe()
    print("\nError summary (pixels):")
    print(summary)


# -----------------------------------------------------------------------------
# NEW: choose prediction by matching tail–rostrum pixel distance
# -----------------------------------------------------------------------------


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Return Euclidean distance between two 2-D points."""

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def choose_best_match_by_length(
    preds: List[Tuple[float, float, float, float, List[Tuple[float, float, float]]]],
    gt_pixel_length: float,
    img_w: int,
    img_h: int,
) -> Tuple[
    Tuple[float, float, float, float, List[Tuple[float, float, float]]],  # best pred
    float,  # predicted pixel length
    float,  # absolute error (pixels)
]:
    """Select the prediction whose tail→rostrum pixel length is closest to the ground truth."""

    best_pred: Tuple[float, float, float, float, List[Tuple[float, float, float]]] | None = None
    best_len_pred: float | None = None
    min_diff = float("inf")

    for pred in preds:
        xc, yc, w_n, h_n, kps = pred

        # Need at least 4 keypoints (start_carapace, eyes, rostrum, tail)
        if len(kps) < 4:
            continue

        rostrum_kp = kps[2]
        tail_kp = kps[3]

        # Convert to pixel space (current image resolution)
        r_px = (rostrum_kp[0] * img_w, rostrum_kp[1] * img_h)
        t_px = (tail_kp[0] * img_w, tail_kp[1] * img_h)

        pixel_length_pred_current = euclidean_distance(r_px, t_px)

        # Scale to reference resolution (5312×2988)
        scale_factor_w = 5312.0 / img_w
        scale_factor_h = 2988.0 / img_h

        # If the image was resized isotropically, both factors are (nearly) equal.
        # We take the average just in case there is a tiny rounding difference.
        scale_factor = (scale_factor_w + scale_factor_h) / 2.0

        pixel_length_pred_scaled = pixel_length_pred_current * scale_factor
        diff = abs(pixel_length_pred_scaled - gt_pixel_length)

        if diff < min_diff:
            min_diff = diff
            best_pred = pred
            best_len_pred = pixel_length_pred_scaled

    # Fallback: if none meet criteria (e.g., missing keypoints), use first pred
    if best_pred is None:
        best_pred = preds[0]
        # compute length for it for consistency
        kps = best_pred[4]
        if len(kps) >= 4:
            r_px = (kps[2][0] * img_w, kps[2][1] * img_h)
            t_px = (kps[3][0] * img_w, kps[3][1] * img_h)
            pixel_length_pred_current = euclidean_distance(r_px, t_px)
            scale_factor_w = 5312.0 / img_w
            scale_factor_h = 2988.0 / img_h
            scale_factor = (scale_factor_w + scale_factor_h) / 2.0
            best_len_pred = pixel_length_pred_current * scale_factor
            min_diff = abs(best_len_pred - gt_pixel_length)
        else:
            best_len_pred = math.nan
            min_diff = math.nan

    return best_pred, best_len_pred, min_diff


if __name__ == "__main__":
    main() 