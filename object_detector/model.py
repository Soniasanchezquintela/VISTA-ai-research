from pathlib import Path
from ultralytics import YOLO

# Paths

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "sku110k_768_e20_pat5.pt"

# Inference parameters
IMG_SIZE = 768          # use the same size you trained with, if possible
# If you see too many false positives, increase CONF_THRES.
CONF_THRES = 0.25
# If neighboring products are being suppressed, increase IOU_THRES slightly, for example to 0.70.
IOU_THRES = 0.60
# shelves can contain many products
MAX_DET = 100


def filter_border_boxes(boxes, img_width, img_height, border = 20):
    """Return only boxes that do not touch the image border."""
    filtered = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        if x1 <= border or y1 <= border or x2 >= img_width - border or y2 >= img_height - border:
            continue
        filtered.append(box)
    return filtered

class ObjectDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model = YOLO(model_path)

    def _detect(self, source, enable_filter: bool, verbose: bool = True) -> tuple[list, any]:
        results = self.model.predict(
            source=source,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_DET,
            save=False,
            verbose=False,
        )

        # results is a list; for one image, take results[0]
        result = results[0]

        # Print detection summary
        num_boxes = len(result.boxes)
        if verbose:
            print(f"Detected products: {num_boxes}")

        if enable_filter:
            result.boxes = filter_border_boxes(result.boxes, img_width=result.orig_img.shape[1], img_height=result.orig_img.shape[0])
            if verbose:
                print(f"Detected products (after filtering border boxes): {len(result.boxes)}")

        # Draw boxes on the image
        annotated = result.plot()

        return result.boxes, annotated

    def detect_from_file(self, image_file: str, enable_filter: bool = True) -> tuple[list, any]:
        image_path = Path(image_file)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return self._detect(str(image_path), enable_filter=enable_filter)

    def detect_from_frame(self, frame, enable_filter: bool = True, verbose: bool = True) -> tuple[list, any]:
        return self._detect(frame, enable_filter=enable_filter, verbose=verbose)


    def print_boxes(self, boxes):
        # Print each box
        for i, box in enumerate(boxes):
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
