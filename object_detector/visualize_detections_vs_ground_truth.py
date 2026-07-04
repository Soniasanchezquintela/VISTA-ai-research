#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2

try:
    from object_detector.model import DEFAULT_MODEL_PATH, ObjectDetector
except ModuleNotFoundError:
    from model import DEFAULT_MODEL_PATH, ObjectDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DETECTOR_COLOR = (255, 180, 0)
GROUND_TRUTH_COLOR = (40, 220, 70)
TEXT_COLOR = (255, 255, 255)
HEADER_BG = (35, 35, 35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side images comparing detector bounding boxes with "
            "ground-truth boxes from the paired JSON annotation files."
        )
    )
    parser.add_argument(
        "--dataset",
        default="dataset",
        type=Path,
        help="Directory containing image/json pairs. Defaults to ./dataset.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional path to a single image to process instead of the full dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("detection_gt_comparison"),
        type=Path,
        help="Directory where comparison images will be written.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        type=Path,
        help=f"YOLO model path. Defaults to {DEFAULT_MODEL_PATH}.",
    )
    parser.add_argument(
        "--max-panel-width",
        default=1200,
        type=int,
        help=(
            "Resize each panel to at most this width before composing. "
            "Use 0 to keep original resolution."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of images to process.",
    )
    parser.add_argument(
        "--no-border-filter",
        action="store_true",
        help="Disable the detector's border-box filtering.",
    )
    return parser.parse_args()


def iter_images(dataset_dir: Path, image: Path | None, limit: int | None) -> Iterable[Path]:
    if image is not None:
        yield image
        return

    images = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        images = images[:limit]

    yield from images


def load_ground_truth(json_path: Path) -> tuple[list[dict], dict[int, dict]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    classes = {
        index: class_info for index, class_info in enumerate(data.get("classes", []))
    }
    return data.get("boxes", []), classes


def maybe_resize(image, max_width: int):
    if max_width <= 0 or image.shape[1] <= max_width:
        return image, 1.0

    scale = max_width / image.shape[1]
    width = int(round(image.shape[1] * scale))
    height = int(round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized, scale


def draw_label(image, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    top = max(0, y - text_height - baseline - 6)
    right = min(image.shape[1] - 1, x + text_width + 8)
    cv2.rectangle(image, (x, top), (right, y), color, -1)
    cv2.putText(
        image,
        text,
        (x + 4, y - baseline - 3),
        font,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_box(
    image,
    bbox: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    x1, y1, x2, y2 = bbox
    pt1 = (int(round(x1 * scale)), int(round(y1 * scale)))
    pt2 = (int(round(x2 * scale)), int(round(y2 * scale)))
    thickness = max(2, int(round(3 * scale)))
    cv2.rectangle(image, pt1, pt2, color, thickness)
    draw_label(image, label, pt1[0], max(pt1[1], 20), color)


def draw_ground_truth_boxes(image, boxes: list[dict], classes: dict[int, dict], scale: float) -> None:
    for index, box in enumerate(boxes, start=1):
        x1 = float(box["x"])
        y1 = float(box["y"])
        x2 = x1 + float(box["w"])
        y2 = y1 + float(box["h"])
        class_id = int(box.get("class", -1))
        class_name = classes.get(class_id, {}).get("name", f"class_{class_id}")
        draw_box(
            image,
            (x1, y1, x2, y2),
            f"{index}: {class_name}",
            GROUND_TRUTH_COLOR,
            scale,
        )


def draw_detector_boxes(image, detections, scale: float) -> None:
    for index, detection in enumerate(detections, start=1):
        draw_box(
            image,
            detection.bbox,
            f"{index}: {detection.confidence:.2f}",
            DETECTOR_COLOR,
            scale,
        )


def add_header(image, text: str):
    header_height = 56
    header = cv2.copyMakeBorder(
        image,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=HEADER_BG,
    )
    cv2.putText(
        header,
        text,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return header


def compose_side_by_side(left, right):
    target_height = max(left.shape[0], right.shape[0])
    left = cv2.copyMakeBorder(
        left,
        0,
        target_height - left.shape[0],
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=HEADER_BG,
    )
    right = cv2.copyMakeBorder(
        right,
        0,
        target_height - right.shape[0],
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=HEADER_BG,
    )
    separator = 255 * (left[:, :8] * 0 + 1)
    return cv2.hconcat([left, separator, right])


def create_comparison(
    image_path: Path,
    json_path: Path,
    detector: ObjectDetector,
    output_dir: Path,
    enable_filter: bool,
    max_panel_width: int,
) -> Path:
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    detections, _ = detector.detect_from_file(str(image_path), enable_filter=enable_filter)
    gt_boxes, classes = load_ground_truth(json_path)

    detector_panel, detector_scale = maybe_resize(original.copy(), max_panel_width)
    ground_truth_panel, ground_truth_scale = maybe_resize(original.copy(), max_panel_width)

    draw_detector_boxes(detector_panel, detections, detector_scale)
    draw_ground_truth_boxes(ground_truth_panel, gt_boxes, classes, ground_truth_scale)

    detector_panel = add_header(
        detector_panel, f"Detector: {len(detections)} boxes"
    )
    ground_truth_panel = add_header(
        ground_truth_panel, f"Ground truth: {len(gt_boxes)} boxes"
    )

    comparison = compose_side_by_side(detector_panel, ground_truth_panel)
    output_path = output_dir / f"{image_path.stem}_detector_vs_gt.jpg"
    cv2.imwrite(str(output_path), comparison)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector = ObjectDetector(model_path=str(args.model))
    enable_filter = not args.no_border_filter

    written = []
    for image_path in iter_images(args.dataset, args.image, args.limit):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            print(f"Skipping {image_path}: missing {json_path.name}")
            continue

        output_path = create_comparison(
            image_path=image_path,
            json_path=json_path,
            detector=detector,
            output_dir=args.output_dir,
            enable_filter=enable_filter,
            max_panel_width=args.max_panel_width,
        )
        written.append(output_path)
        print(f"Wrote {output_path}")

    print(f"Done. Wrote {len(written)} comparison images to {args.output_dir}")


if __name__ == "__main__":
    main()
