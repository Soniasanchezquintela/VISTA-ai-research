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

def sanity_check(dataset: str) -> bool:
    """
    Check if the dataset path exists and contains at least one image file.
    """
    dataset_path = Path(dataset)
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

    if not sanity_check(args.dataset):
        return

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return
    
    # Loop through all images in the dataset directory
    for image_path in dataset_path.glob("*.jpg"):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: Could not read image '{image_path}'. Skipping.")
            continue

        # Load its associated json file with the labeled data
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            print(f"Error: JSON file '{json_path}' does not exist. Skipping.")
            continue

        print(f"Processing image: {image_path} with data from: {json_path}")


    # print("Loading required modules...", flush=True)

    # from object_detector import ObjectDetector
    # from hand_detector import HandDetector
    # from object_identifier import ObjectIdentifier

    # detector = ObjectDetector()
    # hand_detector = HandDetector(mode=HandDetector.Mode.IMAGE)
    # object_identifier = ObjectIdentifier()

    # detections, hand_detection, identifications = execute_pipeline(frame, detector, hand_detector, object_identifier)

if __name__ == "__main__":
    main()
