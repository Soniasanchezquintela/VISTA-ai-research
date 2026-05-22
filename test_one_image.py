#!/usr/bin/env python3

from pathlib import Path
import argparse
from ultralytics import YOLO
import cv2


# Paths
MODEL_PATH = "sku110k_768_e20_pat5.pt"

# Inference parameters
IMG_SIZE = 768          # use the same size you trained with, if possible
# If you see too many false positives, increase CONF_THRES.
CONF_THRES = 0.25
# If neighboring products are being suppressed, increase IOU_THRES slightly, for example to 0.70.
IOU_THRES = 0.60
# shelves can contain many products
MAX_DET = 100

# Output folder
OUTPUT_DIR = Path("inference_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on one shelf image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image, for example /path/to/IMG_3189.jpg",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load trained model
    model = YOLO(MODEL_PATH)

    # Run inference on one image
    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        save=False,
        verbose=True,
    )

    # results is a list; for one image, take results[0]
    result = results[0]

    # Print detection summary
    num_boxes = len(result.boxes)
    print(f"Image: {image_path.name}")
    print(f"Detected products: {num_boxes}")

    # Print each box
    for i, box in enumerate(result.boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        x1, y1, x2, y2 = xyxy
        print(
            f"Box {i + 1}: "
            f"class={cls_id}, "
            f"conf={conf:.3f}, "
            f"x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}"
        )

    # Draw boxes on the image
    annotated = result.plot()

    # Save annotated image
    output_path = OUTPUT_DIR / f"{image_path.stem}_pred.jpg"
    cv2.imwrite(str(output_path), annotated)

    print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    main()

