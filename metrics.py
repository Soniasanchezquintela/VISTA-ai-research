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

def sanity_check(dataset_path: Path) -> bool:
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

def main():

    args = parse_args()

    dataset_path = Path(args.dataset)

    if not sanity_check(dataset_path):
        return

    print("Loading required modules...", flush=True)

    from object_detector import ObjectDetector
    from object_identifier import ObjectIdentifier

    object_detector = ObjectDetector()
    object_identifier = ObjectIdentifier()
   
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
        

    # detections, hand_detection, identifications = execute_pipeline(frame, detector, hand_detector, object_identifier)

if __name__ == "__main__":
    main()
