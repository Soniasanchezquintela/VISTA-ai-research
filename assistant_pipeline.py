#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2

from intent_classifier.classifier import HybridIntentClassifier, IntentClassifier
from intent_classifier.llm_parser import DEFAULT_LLM_MODEL, LLMIntentParser
from intent_classifier.schema import Intent, IntentResult


DEFAULT_BERT_CHECKPOINT = Path("intent_classifier/checkpoints/intent_classifier.pt")


def normalize_for_match(text: str | None) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9ñ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str | None) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "de",
        "del",
        "el",
        "en",
        "for",
        "la",
        "las",
        "le",
        "los",
        "of",
        "on",
        "the",
        "un",
        "una",
        "y",
    }
    return {
        token
        for token in normalize_for_match(text).split()
        if len(token) > 1 and token not in stopwords
    }


def product_text(record: dict) -> str:
    product = record.get("product") or {}
    fields = [
        product.get("description"),
        product.get("category"),
        product.get("sku_id"),
    ]
    return " ".join(field for field in fields if field)


def product_name(record: dict) -> str:
    product = record.get("product") or {}
    if product.get("is_known") and product.get("description"):
        return product["description"]
    if product.get("description"):
        return f"possible match: {product['description']}"
    return "the detected product"


def scene_product_name(record: dict) -> str | None:
    product = record.get("product") or {}
    if product.get("is_known") and product.get("description"):
        return product["description"]
    return None


def position_text(record: dict) -> str:
    shelf_position = record.get("shelf_position") or {}
    if shelf_position.get("phrase"):
        phrase = shelf_position["phrase"]
        order = shelf_position.get("order_on_shelf")
        if order is not None:
            return f"{phrase}, around item {order} from the left"
        return phrase

    position = record.get("position") or {}
    return position.get("phrase", "the detected shelf area")


def clamp_box_to_image(box_xyxy, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, box_xyxy)
    x1 = max(0, min(x1, image_width))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height))
    y2 = max(0, min(y2, image_height))
    return x1, y1, x2, y2


def describe_box_position(
    box_xyxy: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> dict:
    x1, y1, x2, y2 = box_xyxy
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    horizontal = describe_horizontal_position(center_x, image_width)
    vertical_index = min(2, max(0, int(center_y / (image_height / 3))))
    vertical = ["top", "middle", "bottom"][vertical_index]

    if horizontal == "center":
        phrase = f"{vertical} shelf area"
    elif vertical == "middle":
        phrase = f"middle {horizontal} shelf area"
    else:
        phrase = f"{vertical} {horizontal} shelf area"

    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "phrase": phrase,
        "center": {
            "x": round(center_x, 2),
            "y": round(center_y, 2),
            "x_norm": round(center_x / image_width, 4),
            "y_norm": round(center_y / image_height, 4),
        },
    }


def make_detection_record(
    index: int,
    box,
    image_width: int,
    image_height: int,
    identification: dict | None = None,
    min_product_score: float = 0.70,
    min_product_confidence: float = 0.80,
) -> dict:
    x1, y1, x2, y2 = clamp_box_to_image(
        box.xyxy[0].cpu().numpy(),
        image_width=image_width,
        image_height=image_height,
    )
    width = x2 - x1
    height = y2 - y1

    record = {
        "index": index,
        "bbox_xyxy": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
        "bbox_normalized_xyxy": {
            "x1": round(x1 / image_width, 4),
            "y1": round(y1 / image_height, 4),
            "x2": round(x2 / image_width, 4),
            "y2": round(y2 / image_height, 4),
        },
        "size": {
            "width": width,
            "height": height,
            "area": width * height,
        },
        "position": describe_box_position((x1, y1, x2, y2), image_width, image_height),
    }

    if hasattr(box, "conf") and box.conf is not None:
        record["confidence"] = round(float(box.conf[0].cpu().numpy()), 4)

    if hasattr(box, "cls") and box.cls is not None:
        record["class_id"] = int(box.cls[0].cpu().numpy())

    if identification is not None:
        product = identification.get("product", {})
        record["product"] = {
            "description": product.get("description", "unknown"),
            "category": product.get("category", "unknown"),
            "sku_id": identification.get("sku_id"),
            "score": round(float(identification.get("score", 0.0)), 4),
            "confidence": round(float(identification.get("confidence", 0.0)), 4),
            "is_known": (
                identification.get("score", 0.0) >= min_product_score
                and identification.get("confidence", 0.0) >= min_product_confidence
            ),
        }

    return record


def estimate_shelf_cluster_threshold(records: list[dict], image_height: int) -> float:
    if not records:
        return 0.0

    heights = sorted(record["size"]["height"] for record in records)
    median_height = heights[len(heights) // 2]
    return max(40.0, min(median_height * 0.50, image_height * 0.08))


def cluster_records_into_shelves(
    records: list[dict],
    image_width: int,
    image_height: int,
    threshold: float | None = None,
) -> list[dict]:
    if not records:
        return []

    cluster_threshold = threshold
    if cluster_threshold is None:
        cluster_threshold = estimate_shelf_cluster_threshold(records, image_height)

    sorted_records = sorted(records, key=lambda record: record["position"]["center"]["y"])
    clusters = []

    for record in sorted_records:
        center_y = record["position"]["center"]["y"]
        if not clusters:
            clusters.append({"records": [record], "center_y": center_y})
            continue

        nearest_cluster = min(
            clusters,
            key=lambda cluster: abs(center_y - cluster["center_y"]),
        )
        if abs(center_y - nearest_cluster["center_y"]) <= cluster_threshold:
            nearest_cluster["records"].append(record)
            nearest_cluster["center_y"] = sum(
                item["position"]["center"]["y"] for item in nearest_cluster["records"]
            ) / len(nearest_cluster["records"])
        else:
            clusters.append({"records": [record], "center_y": center_y})

    clusters.sort(key=lambda cluster: cluster["center_y"])
    shelf_count = len(clusters)
    shelf_summaries = []

    for shelf_index, cluster in enumerate(clusters, start=1):
        shelf_records = sorted(
            cluster["records"],
            key=lambda record: record["position"]["center"]["x"],
        )
        y_values = [record["position"]["center"]["y"] for record in shelf_records]
        y_min = min(record["bbox_xyxy"]["y1"] for record in shelf_records)
        y_max = max(record["bbox_xyxy"]["y2"] for record in shelf_records)

        shelf_label = describe_shelf_label(shelf_index, shelf_count)
        shelf_summary = {
            "shelf_index": shelf_index,
            "shelf_count": shelf_count,
            "label": shelf_label,
            "center_y": round(sum(y_values) / len(y_values), 2),
            "center_y_norm": round((sum(y_values) / len(y_values)) / image_height, 4),
            "y_range": {
                "y1": y_min,
                "y2": y_max,
            },
            "detection_count": len(shelf_records),
        }
        shelf_summaries.append(shelf_summary)

        for order_on_shelf, record in enumerate(shelf_records, start=1):
            horizontal = describe_horizontal_position(
                record["position"]["center"]["x"],
                image_width,
            )
            record["shelf_position"] = {
                **shelf_summary,
                "horizontal": horizontal,
                "order_on_shelf": order_on_shelf,
                "phrase": f"{shelf_label}, {horizontal} side",
            }

    return shelf_summaries


def describe_shelf_label(shelf_index: int, shelf_count: int) -> str:
    if shelf_count == 1:
        return "the only detected shelf"
    if shelf_index == 1:
        return "top shelf"
    if shelf_index == shelf_count:
        return "bottom shelf"
    return f"shelf {shelf_index} from the top"


def describe_horizontal_position(center_x: float, image_width: int) -> str:
    horizontal_index = min(2, max(0, int(center_x / (image_width / 3))))
    return ["left", "center", "right"][horizontal_index]


def summarize_product_counts(names: list[str], unknown_count: int) -> str:
    parts = []
    counts = Counter(names)
    for name, count in counts.most_common():
        if count == 1:
            parts.append(name)
        else:
            parts.append(f"{count} {name}")

    if unknown_count:
        if unknown_count == 1:
            parts.append("1 unidentified product")
        else:
            parts.append(f"{unknown_count} unidentified products")

    if not parts:
        return "no identified products"

    if len(parts) == 1:
        return parts[0]

    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def shelf_matches_constraint(record: dict, constraint: str | None) -> bool:
    if not constraint:
        return True

    constraint = normalize_for_match(constraint).replace(" ", "_")
    shelf_position = record.get("shelf_position") or {}
    label = normalize_for_match(shelf_position.get("label"))
    horizontal = normalize_for_match(shelf_position.get("horizontal"))
    shelf_index = shelf_position.get("shelf_index")
    shelf_count = shelf_position.get("shelf_count")

    if constraint in {"left", "center", "right"}:
        return horizontal == constraint

    if constraint in {"top", "upper"}:
        return shelf_index == 1 or "top" in label

    if constraint in {"bottom", "lower"}:
        return shelf_index == shelf_count or "bottom" in label

    if constraint == "middle":
        if shelf_index is None or shelf_count is None:
            return "middle" in label
        return shelf_index not in {1, shelf_count}

    if constraint == "top_left":
        return shelf_matches_constraint(record, "top") and shelf_matches_constraint(record, "left")

    if constraint == "top_right":
        return shelf_matches_constraint(record, "top") and shelf_matches_constraint(record, "right")

    if constraint == "bottom_left":
        return shelf_matches_constraint(record, "bottom") and shelf_matches_constraint(record, "left")

    if constraint == "bottom_right":
        return shelf_matches_constraint(record, "bottom") and shelf_matches_constraint(record, "right")

    return True


def score_target_match(record: dict, target: str | None) -> float:
    if not target:
        return 0.0

    target_tokens = tokenize(target)
    candidate_tokens = tokenize(product_text(record))
    if not target_tokens or not candidate_tokens:
        return 0.0

    overlap = target_tokens & candidate_tokens
    if not overlap:
        return 0.0

    return len(overlap) / len(target_tokens)


def find_best_match(records: Iterable[dict], target: str | None, shelf_constraint: str | None) -> dict | None:
    candidates = [
        record
        for record in records
        if shelf_matches_constraint(record, shelf_constraint)
    ]
    scored = [
        (score_target_match(record, target), record)
        for record in candidates
    ]
    scored = [(score, record) for score, record in scored if score > 0]
    if not scored:
        return None

    scored.sort(
        key=lambda item: (
            item[0],
            item[1].get("product", {}).get("is_known", False),
            item[1].get("product", {}).get("score", 0.0),
        ),
        reverse=True,
    )
    return scored[0][1]


def describe_scene(detections: dict) -> str:
    shelves = detections.get("shelves") or []
    records = detections.get("detections") or []
    if not records:
        return "I do not see any products clearly enough to describe."

    if not shelves:
        return f"I detected {len(records)} product-like objects, but I could not group them into shelves."

    shelf_parts = []
    for shelf in shelves:
        shelf_records = [
            record
            for record in records
            if (record.get("shelf_position") or {}).get("shelf_index") == shelf["shelf_index"]
        ]
        names = [
            name
            for name in (scene_product_name(record) for record in shelf_records)
            if name is not None
        ]
        unknown_count = len(shelf_records) - len(names)
        shelf_parts.append(
            f"{shelf['label']}: {summarize_product_counts(names, unknown_count)}"
        )

    return "I detected " + str(len(shelves)) + " shelves. " + "; ".join(shelf_parts) + "."


def answer_from_intent(intent_result: IntentResult, detections: dict) -> str:
    records = detections.get("detections") or []

    if intent_result.intent == Intent.DESCRIBE_SCENE:
        return describe_scene(detections)

    if intent_result.intent in {
        Intent.NAVIGATE_TO_TARGET,
        Intent.CONFIRM_TARGET_PRESENT,
        Intent.GET_PRICE,
    }:
        if not intent_result.target:
            return "Which product should I look for?"

        match = find_best_match(
            records,
            target=intent_result.target,
            shelf_constraint=intent_result.shelf_constraint,
        )

        if match is None:
            unconstrained_match = find_best_match(
                records,
                target=intent_result.target,
                shelf_constraint=None,
            )
            if unconstrained_match is not None and intent_result.shelf_constraint:
                name = product_name(unconstrained_match)
                position = position_text(unconstrained_match)
                return (
                    f"I did not find {intent_result.target} on the "
                    f"{intent_result.shelf_constraint} area, but I found {name} "
                    f"on the {position}."
                )

            constraint = f" on the {intent_result.shelf_constraint} area" if intent_result.shelf_constraint else ""
            return f"I could not find {intent_result.target}{constraint}."

        name = product_name(match)
        position = position_text(match)

        if intent_result.intent == Intent.CONFIRM_TARGET_PRESENT:
            return f"Yes, I found {name} on the {position}."

        if intent_result.intent == Intent.GET_PRICE:
            return f"I found {name} on the {position}, but I do not have price reading connected yet."

        return f"I found {name} on the {position}."

    if intent_result.intent == Intent.DESCRIBE_POINTED_PRODUCT:
        return "Pointed-product description is not connected yet. The next step is to use the hand direction to select one detected box."

    if intent_result.intent == Intent.READ_TEXT:
        return "Text reading is not connected yet. The next step is to add OCR for product labels and shelf tags."

    return "Sorry, I did not understand the request."


def run_vision_on_image(
    image_path: Path,
    identify_products: bool,
    shelf_cluster_threshold: float | None,
) -> dict:
    from object_detector import ObjectDetector

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot open image file: {image_path}")

    detector = ObjectDetector()
    boxes, _ = detector.detect_from_frame(image)

    object_identifier = None
    if identify_products:
        from object_identifier import ObjectIdentifier

        object_identifier = ObjectIdentifier()

    height, width = image.shape[:2]
    records = []

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = clamp_box_to_image(
            box.xyxy[0].cpu().numpy(),
            image_width=width,
            image_height=height,
        )
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        identification = None
        if object_identifier is not None:
            identification = object_identifier.identify_product(crop)

        records.append(
            make_detection_record(index, box, width, height, identification)
        )

    shelves = cluster_records_into_shelves(
        records,
        image_width=width,
        image_height=height,
        threshold=shelf_cluster_threshold,
    )

    return {
        "image": str(image_path),
        "image_size": {
            "width": width,
            "height": height,
        },
        "shelf_count": len(shelves),
        "shelves": shelves,
        "detections": records,
    }


def load_or_create_detections(args) -> dict:
    if args.detections_json is not None:
        return json.loads(args.detections_json.read_text(encoding="utf-8"))

    if args.image is None:
        raise ValueError("Provide either --image or --detections-json.")

    detections = run_vision_on_image(
        image_path=args.image,
        identify_products=not args.skip_identification,
        shelf_cluster_threshold=args.shelf_cluster_threshold,
    )
    if args.output_detections_json is not None:
        args.output_detections_json.write_text(
            json.dumps(detections, indent=2),
            encoding="utf-8",
        )
        print(f"Saved detections to: {args.output_detections_json}")

    return detections


def make_intent_parser(args):
    if args.intent_backend == "llm":
        return LLMIntentParser(
            model=args.llm_model,
            confidence_threshold=args.llm_confidence_threshold,
        )

    bert_classifier = IntentClassifier(
        checkpoint_path=args.checkpoint,
        confidence_threshold=args.bert_confidence_threshold,
        device=args.device,
    )
    if args.intent_backend == "bert":
        return bert_classifier

    llm_parser = LLMIntentParser(
        model=args.llm_model,
        confidence_threshold=args.llm_confidence_threshold,
    )
    return HybridIntentClassifier(
        bert_classifier=bert_classifier,
        llm_parser=llm_parser,
        llm_fallback_threshold=args.llm_fallback_threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link intent parsing with shelf product detections and produce a spoken-style answer."
    )
    parser.add_argument("question", help="User question or command.")
    parser.add_argument("--image", type=Path, help="Shelf image to process with YOLO and product identification.")
    parser.add_argument("--detections-json", type=Path, help="Existing detections JSON from project.py.")
    parser.add_argument(
        "--output-detections-json",
        type=Path,
        help="When using --image, save detections so future questions can use --detections-json.",
    )
    parser.add_argument(
        "--intent-backend",
        choices=("bert", "llm", "hybrid"),
        default="llm",
        help="Intent parser to use. LLM is the easiest option before the BERT checkpoint exists.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_BERT_CHECKPOINT)
    parser.add_argument("--bert-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-confidence-threshold", type=float, default=0.45)
    parser.add_argument("--llm-fallback-threshold", type=float, default=0.70)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-identification", action="store_true")
    parser.add_argument("--shelf-cluster-threshold", type=float, default=None)
    parser.add_argument(
        "--print-intent",
        action="store_true",
        help="Print parsed intent details before the assistant reply.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intent_parser = make_intent_parser(args)
    intent_result = intent_parser.predict(args.question)
    detections = load_or_create_detections(args)
    answer = answer_from_intent(intent_result, detections)

    if args.print_intent:
        print(f"intent: {intent_result.intent.value}")
        print(f"target: {intent_result.target}")
        print(f"confidence: {intent_result.confidence:.3f}")
        print(f"source: {intent_result.source}")
        if intent_result.shelf_constraint:
            print(f"shelf_constraint: {intent_result.shelf_constraint}")
        print()

    print(answer)


if __name__ == "__main__":
    main()
