from pathlib import Path
import argparse
from object_detector import ObjectDetector
import cv2
from hand_detector import HandDetector
from object_identifier import ObjectIdentifier

"""
def handle_user_command(text: str) -> None:
    result = intent_classifier.predict(text)

    if result.intent == Intent.DESCRIBE_SCENE:
        describe_scene()

    elif result.intent == Intent.DESCRIBE_POINTED_PRODUCT:
        describe_pointed_product()

    elif result.intent == Intent.NAVIGATE_TO_TARGET:
        if result.target is None:
            ask_user_for_target()
        else:
            navigate_to_target(result.target)

    elif result.intent == Intent.CONFIRM_TARGET_PRESENT:
        if result.target is None:
            ask_user_for_target()
        else:
            confirm_target_present(result.target)

    elif result.intent == Intent.GET_PRICE:
        get_price_of_pointed_product()

    elif result.intent == Intent.READ_TEXT:
        read_visible_text()

    else:
        say("Sorry, I did not understand.")
"""


def annotate_box(frame, boxes, index: int, color):
    annotated_frame = frame
    frame_height, frame_width = annotated_frame.shape[:2]
    line_thickness = max(3, min(frame_height, frame_width) // 200)
    font_scale = max(1.2, min(frame_height, frame_width) / 500)
    font_thickness = max(3, line_thickness)

    box = boxes[index]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_thickness)

    label = str(index)
    (label_width, label_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )
    label_top = max(0, y1 - label_height - baseline - 16)
    label_bottom = label_top + label_height + baseline + 12
    label_right = x1 + label_width + 20

    cv2.rectangle(
        annotated_frame,
        (x1, label_top),
        (label_right, label_bottom),
        color,
        -1,
    )

    label_origin = (x1 + 10, label_bottom - baseline - 6)
    cv2.putText(
        annotated_frame,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA,
    )

    return annotated_frame


def execute_pipeline(frame, object_detector, hand_detector):
    # Run object detection
    boxes, annotated_frame = object_detector.detect_from_frame(frame)

    # Run hand detection
    found, start_point, direction, hand_annotated_frame = hand_detector.detect(frame)

    # Combine annotations if both detections are successful
    if annotated_frame is not None and hand_annotated_frame is not None:
        combined_frame = cv2.addWeighted(annotated_frame, 0.5, hand_annotated_frame, 0.5, 0)
        return combined_frame
    elif annotated_frame is not None:
        return annotated_frame
    elif hand_annotated_frame is not None:
        return hand_annotated_frame
    else:
        return frame

def process_image(image_path: str, save: bool = False, extract_boxes: bool = False) -> int:
    detector = ObjectDetector()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot open image file: {image_path}")

    boxes, _ = detector.detect_from_frame(image)

    #hand_detector = HandDetector(mode=HandDetector.Mode.IMAGE)
    found = False
    #found, _, _, hand_annotated_image = hand_detector.detect_from_file(image_path)
    object_identifier = ObjectIdentifier()


    preview_image = image.copy()

   
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Convert to int
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Optional but recommended: clamp coordinates to image size
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            print(f"Skipping empty crop for box {i}: {(x1, y1, x2, y2)}")
            continue

        # send cropped image to object identifier
        result_identification = object_identifier.identify_product(crop)

        product = result_identification["product"]

        description = product.get("description", "unknown")
        category = product.get("category", "unknown")

        if result_identification["score"] < ObjectIdentifier.MIN_SCORE or result_identification["confidence"] < ObjectIdentifier.MIN_CONFIDENCE:
            annotate_box(preview_image, boxes, i, (0, 0, 255))
            print(f"Cropped Image: {i}")
            print("Unknown")
            print(f"Best SKU ID: {result_identification['sku_id']}")
            print(f"Best reference image: {result_identification['reference_image_path']}")
            print(f"Best score: {result_identification['score']:.4f}")
            print(f"Best confidence: {result_identification['confidence']:.4f}")
            print("-" * 60)
            continue

        annotate_box(preview_image, boxes, i, (0, 255, 0))
        print(f"Cropped Image: {i}")
        print(f"SKU ID:     {result_identification['sku_id']}")
        print(f"Ref image:  {result_identification['reference_image_id']}")
        print(f"Match:      {description}")
        print(f"Category:   {category}")
        print(f"Score:      {result_identification['score']:.4f}")
        print(f"Confidence: {result_identification['confidence']:.4f}")
        print("-" * 60)

        if extract_boxes:
            # Save bounding boxes one by one to a new image file
    
            output_path = Path(f"{Path(image_path).stem}_box_{i}.jpg")
            cv2.imwrite(str(output_path), crop)

            print(f"Saved bounding box {i} to: {output_path}")       

    if save:
        # Save annotated image
        output_path = Path(f"{Path(image_path).stem}_pred.jpg")
        cv2.imwrite(str(output_path), preview_image)
        print(f"Saved annotated image to: {output_path}")

    if preview_image is not None:
        window_name = "Hand Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, preview_image)
        if found:
            print("Hand detected in image.")
        else:
            print("No hand detected in image.")
        print("Press any key to close the hand preview window.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


    return 0

def process_video(video_path: str, save: bool = False) -> int:
    detector = ObjectDetector()
    hand_detector = HandDetector(mode=HandDetector.Mode.VIDEO)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define video writer
    out = None
    if save:
        output_path = Path(f"{Path(video_path).stem}_pred.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Video: {Path(video_path).name}")
    print(f"FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")
    print("Processing video...")
    print("Press 'q' to close the preview window.")

    cv2.namedWindow("Annotated Video", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference on frame
        #boxes, annotated_frame = detector.detect_from_frame(frame)
        found, start_point, direction, hand_annotated_frame = hand_detector.detect(frame, timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        
        if found and hand_annotated_frame is not None:
            annotated_frame = hand_annotated_frame
            cv2.imshow("Annotated Video", annotated_frame)
        else:
            cv2.imshow("Annotated Video", frame)
        
        # Show the frame
        #cv2.imshow("Annotated Video", annotated_frame)

        # Process GUI events so the window updates and can receive key presses
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview stopped by user.")
            break
        
        # Write frame to output video
        if save and out:
            out.write(annotated_frame)
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    return 0

def process_webcam():
    print("Webcam processing is not implemented yet.")
    return 0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on one shelf image."
    )

    parser.add_argument(
        "--image",
        required=False,
        help="Path to input image, for example /path/to/IMG_3189.jpg",
    )

    parser.add_argument(
        "--extract-boxes",
        action="store_true",
        help="Extract bounding boxes from the image.",
    )

    parser.add_argument(
        "--video",
        required=False,
        help="Path to input video, for example /path/to/video.mp4",
    )

    parser.add_argument(
        "--webcam",
        required=False,
        help="Use webcam for real-time inference.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated image to disk.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    counter = sum([bool(args.image), bool(args.video), bool(args.webcam)])
    if counter > 1:
        print("Error: Please provide only one of --image, --video, or --webcam.")
        exit(1)
    
    if counter == 0:
        print("Error: Please provide one of --image, --video, or --webcam to process.")
        exit(1)

    if args.image:
        exit(process_image(args.image, save=args.save, extract_boxes=args.extract_boxes))

    if args.video:
        exit(process_video(args.video, save=args.save))

    if args.webcam:
        exit(process_webcam())

if __name__ == "__main__":
    main()
