from pathlib import Path
import cv2
import numpy as np


def auto_crop_product(
    image_path: str | Path,
    output_path: str | Path,
    padding: int = 20,
    min_area_ratio: float = 0.01,
) -> tuple[int, int, int, int] | None:
    """
    Auto-crop a single product from a mostly solid background image.

    Returns:
        (x1, y1, x2, y2) bbox in original image coordinates,
        or None if no foreground object is found.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]

    # Estimate background color from the four corners
    corner_size = max(5, min(h, w) // 20)

    corners = np.concatenate([
        image[:corner_size, :corner_size].reshape(-1, 3),
        image[:corner_size, -corner_size:].reshape(-1, 3),
        image[-corner_size:, :corner_size].reshape(-1, 3),
        image[-corner_size:, -corner_size:].reshape(-1, 3),
    ])

    bg_color = np.median(corners, axis=0)

    # Difference from estimated background
    diff = np.linalg.norm(image.astype(np.float32) - bg_color.astype(np.float32), axis=2)

    # Adaptive threshold based on image content
    threshold = max(18, np.percentile(diff, 85) * 0.5)
    mask = (diff > threshold).astype(np.uint8) * 255

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find foreground contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = h * w * min_area_ratio
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        return None

    # Union all relevant contours.
    # This is useful for products with separated parts, labels, caps, shadows, etc.
    all_points = np.vstack(valid_contours)
    x, y, bw, bh = cv2.boundingRect(all_points)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    crop = image[y1:y2, x1:x2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)

    return x1, y1, x2, y2


if __name__ == "__main__":
    input_dir = Path("raw_product_images")
    output_dir = Path("cropped_product_images")

    for image_path in input_dir.glob("*.*"):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        output_path = output_dir / f"{image_path.stem}_crop.jpg"

        bbox = auto_crop_product(
            image_path=image_path,
            output_path=output_path,
            padding=25,
        )

        if bbox is None:
            print(f"FAILED: {image_path}")
        else:
            print(f"OK: {image_path} -> {output_path}, bbox={bbox}")
