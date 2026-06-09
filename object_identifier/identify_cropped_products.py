#!/usr/bin/env python3

import argparse
from pathlib import Path

from clip import ObjectIdentifier

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_image_files(directory: Path):
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify supermarket products from YOLO crop images."
    )

    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing YOLO product crop images.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.03,
        help="Softmax temperature used to estimate confidence.",
    )

    args = parser.parse_args()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {args.image_dir}")

    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.image_dir}")

    image_files = list(iter_image_files(args.image_dir))

    if not image_files:
        print(f"No image files found in: {args.image_dir}")
        return

    object_identifier = ObjectIdentifier()

    print(f"Found {len(image_files)} image file(s).")
    print()

    for image_path in image_files:
        try:
            image = Image.open(image_path).convert("RGB")
            result = object_identifier.identify_product(
                image=image,
                temperature=args.temperature,
            )

            product = result["product"]

            description = product.get("description", "unknown")
            category = product.get("category", "unknown")

            if result["score"] < ObjectIdentifier.MIN_SCORE or result["confidence"] < ObjectIdentifier.MIN_CONFIDENCE:
                print(f"Image: {image_path.name}")
                print("Unknown")
                print(f"Best SKU ID: {result['sku_id']}")
                print(f"Best reference image: {result['reference_image_path']}")
                print(f"Best score: {result['score']:.4f}")
                print(f"Best confidence: {result['confidence']:.4f}")
                print("-" * 60)
                continue
            print(f"Image:      {image_path.name}")
            print(f"SKU ID:     {result['sku_id']}")
            print(f"Ref image:  {result['reference_image_id']}")
            print(f"Match:      {description}")
            print(f"Category:   {category}")
            print(f"Score:      {result['score']:.4f}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("-" * 60)

        except Exception as exc:
            print(f"Image: {image_path.name}")
            print(f"Error: {exc}")
            print("-" * 60)


if __name__ == "__main__":
    main()
