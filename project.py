from pathlib import Path
import argparse
from object_detector import ObjectDetector
import cv2
from hand_detector import HandDetector

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

def process_image(image_path: str, save: bool = False) -> int:
    detector = ObjectDetector()
    boxes, annotated_image = detector.detect_from_file(image_path)

    if save:
        # Save annotated image
        output_path = Path(f"{Path(image_path).stem}_pred.jpg")
        cv2.imwrite(str(output_path), annotated_image)
        print(f"Saved annotated image to: {output_path}")

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
        exit(process_image(args.image, save=args.save))

    if args.video:
        exit(process_video(args.video, save=args.save))

    if args.webcam:
        exit(process_webcam())

if __name__ == "__main__":
    main()
