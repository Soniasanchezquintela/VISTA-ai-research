#! /usr/bin/env python3
from pathlib import Path
import argparse
import cmd
import cv2
from contextlib import contextmanager
import json
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import threading

BASE_DIR = Path(__file__).resolve().parent
IDS_PATH = BASE_DIR / "object_identifier/product_db/embeddings/product_ids.json"

def load_product_index() -> dict[str, int]:
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        product_entries = json.load(f)

    # Product entries will contain duplicates of the sku ids, so we will
    # create a dictionary with the sku id as the key and the number of entries
    # as the value.
    full_products = {}
    for entry in product_entries:
        sku_id = entry["sku_id"]
        if sku_id not in full_products:
            full_products[sku_id] = 0
        full_products[sku_id] += 1

    return full_products

def sanity_check(dataset_path: Path, check_missing_sku: bool = True) -> bool:
    """
    Check if the dataset path exists and contains at least one image file.
    """
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return False

    image_files = list(dataset_path.glob("*.jpg"))
    if not image_files:
        print(f"Error: No image files found in '{dataset_path}'.")
        return False


    json_files_ok = True

    # Loop through all images in the dataset directory
    for image_path in dataset_path.glob("*.jpg"):
        # Load its associated json file with the labeled data
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            print(f"[ERROR] JSON file '{json_path}' does not exist.")
            json_files_ok = False

    if not check_missing_sku:
        return json_files_ok

    sku_ids = load_product_index()

    sku_ids_ok = True
    missing_sku_ids = set()
    # Loop through all json files in the dataset directory and parse them
    for json_path in dataset_path.glob("*.json"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to read or parse JSON file '{json_path}': {e}")
            return False
        else:
            # iterate through all classes in the json file and check that
            # we have them in the product database
            for product_class in data.get("classes", []):
                sku_id = product_class["name"]
                if sku_id not in sku_ids:
                    sku_ids_ok = False
                    missing_sku_ids.add(sku_id)

    if missing_sku_ids:
        for i, sku_id in enumerate(missing_sku_ids):
            print(f"[{i + 1}][ERROR] Missing SKU ID: {sku_id}")

    return sku_ids_ok and json_files_ok

def get_image_json_file(image_path: Path) -> dict:
    """
    Given an image path, return the associated JSON data as a dictionary.
    """
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file '{json_path}' does not exist.")

    with open(json_path, "r") as f:
        data = json.load(f)

    return data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate metrics for object detection and identification."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset directory, for example /path/to/dataset",
    )

    return parser.parse_args()

def _bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """
    Intersection-over-Union between two boxes in (x1, y1, x2, y2) format.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


def _detection_bbox(detection) -> tuple[float, float, float, float]:
    """
    Return a detector bbox from either ProductDetection or a dict-like object.
    """
    bbox = detection["bbox"] if isinstance(detection, dict) else detection.bbox
    x1, y1, x2, y2 = bbox
    return float(x1), float(y1), float(x2), float(y2)


def _ground_truth_bbox(box: dict) -> tuple[float, float, float, float]:
    """
    Convert a labeled JSON box from (x, y, w, h) to (x1, y1, x2, y2).
    """
    x1 = float(box["x"])
    y1 = float(box["y"])
    return x1, y1, x1 + float(box["w"]), y1 + float(box["h"])


def compute_detector_metrics(
    detections: list,
    ground_truth: dict,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute geometric object-detection metrics from detected and labeled boxes.

    Args:
        detections (list): Detected products with a bbox in (x1, y1, x2, y2)
            format. Items can be ProductDetection objects or dictionaries with a
            "bbox" key.
        ground_truth (dict): Ground truth data containing:
            - "boxes": Labeled boxes in (x, y, w, h) format.
        iou_threshold (float): Minimum IoU needed to count a detected box as a
            match for a labeled box.
    """
    detected_boxes = [_detection_bbox(detection) for detection in detections]
    labeled_boxes = [_ground_truth_bbox(box) for box in ground_truth.get("boxes", [])]

    candidate_matches = []
    for detection_index, detected_box in enumerate(detected_boxes):
        for label_index, labeled_box in enumerate(labeled_boxes):
            iou = _bbox_iou(detected_box, labeled_box)
            if iou >= iou_threshold:
                candidate_matches.append((iou, detection_index, label_index))

    candidate_matches.sort(reverse=True)

    matched_detections = set()
    matched_labels = set()

    for iou, detection_index, label_index in candidate_matches:
        if detection_index in matched_detections or label_index in matched_labels:
            continue

        matched_detections.add(detection_index)
        matched_labels.add(label_index)

    true_positives = len(matched_detections)
    false_positives = len(detected_boxes) - true_positives
    false_negatives = len(labeled_boxes) - true_positives

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

def main():

    args = parse_args()

    dataset_path = Path(args.dataset)

    check_missing_sku=False

    if not sanity_check(dataset_path, check_missing_sku=check_missing_sku):
        return

    print("Loading required modules...", flush=True)

    from object_detector import ObjectDetector
    from object_identifier import ObjectIdentifier

    object_detector = ObjectDetector()
    object_identifier = ObjectIdentifier()

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    images_processed = 0
   
    # Loop through all images in the dataset directory
    for image_path in dataset_path.glob("*.jpg"):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image '{image_path}'. Skipping.")
            continue

        # Load its associated json file with the labeled data
        data = get_image_json_file(image_path.with_suffix(".json"))

        print(f"Processing image: {image_path}")
        detections, _ = object_detector.detect_from_frame(frame, verbose=True)

        identifications = []
        if check_missing_sku:
            identifications = object_identifier.identify_boxes(frame, detections, verbose=True)

        print("Number of detections:", len(detections))
        print("Number of identifications:", len(identifications))

        # data["boxes"] contains a list of boxes with the following format:
        # "class": 4,
        # "x": 73.57377049180329,
        # "y": 688.9180327868853,
        # "w": 708.983606557377,
        # "h": 1772.4590163934427
        # Compare these boxes with the detected boxes
        metrics = compute_detector_metrics(detections, data, iou_threshold=0.5)

        total_true_positives += metrics["true_positives"]
        total_false_positives += metrics["false_positives"]
        total_false_negatives += metrics["false_negatives"]
        images_processed += 1

    precision = (
        total_true_positives / (total_true_positives + total_false_positives)
        if total_true_positives + total_false_positives > 0
        else 0.0
    )
    recall = (
        total_true_positives / (total_true_positives + total_false_negatives)
        if total_true_positives + total_false_negatives > 0
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    print("Detector metrics for full dataset:")
    print("Images processed:", images_processed)
    print("True positives:", total_true_positives)
    print("False positives:", total_false_positives)
    print("False negatives:", total_false_negatives)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1_score:.4f}")


    # detections, hand_detection, identifications = execute_pipeline(frame, detector, hand_detector, object_identifier)

if __name__ == "__main__":
    main()
