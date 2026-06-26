#! /usr/bin/env python3
from pathlib import Path
import argparse
import cmd
from contextlib import contextmanager
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import threading

print("Loading required modules...", flush=True)

from object_detector import ObjectDetector
import cv2
from hand_detector import HandDetector
from object_identifier import ObjectIdentifier

from scene_memory import ProductDetection, ProductIdentification
from scene_memory.tracked_scene_memory import TrackedShelfSceneMemory as ShelfSceneMemory

from voice_to_text import VoiceCommandProcessor

from intent_classifier import SimpleIntentClassifier, Intent

if sys.platform == "win32":
    import msvcrt
else:
    import select
    import termios
    import tty

scene_memory = ShelfSceneMemory()
voice_processor = VoiceCommandProcessor()
intent_classifier = SimpleIntentClassifier()


class ThreadOutputRouter:
    """Route stdout/stderr to a file for one thread without affecting others."""

    def __init__(self, fallback):
        self._fallback = fallback
        self._local = threading.local()

    def write(self, text):
        return getattr(self._local, "stream", self._fallback).write(text)

    def flush(self):
        return getattr(self._local, "stream", self._fallback).flush()

    def isatty(self):
        return getattr(self._local, "stream", self._fallback).isatty()

    def __getattr__(self, name):
        return getattr(self._fallback, name)

    @contextmanager
    def redirect(self, stream):
        previous_stream = getattr(self._local, "stream", None)
        self._local.stream = stream
        try:
            yield
        finally:
            if previous_stream is None:
                del self._local.stream
            else:
                self._local.stream = previous_stream


_stdout_router = None
_stderr_router = None


def install_thread_output_routers() -> tuple[ThreadOutputRouter, ThreadOutputRouter]:
    global _stdout_router, _stderr_router

    if _stdout_router is None:
        _stdout_router = ThreadOutputRouter(sys.stdout)
        sys.stdout = _stdout_router
    if _stderr_router is None:
        _stderr_router = ThreadOutputRouter(sys.stderr)
        sys.stderr = _stderr_router

    return _stdout_router, _stderr_router


def is_window_open(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def close_window_if_open(window_name: str) -> None:
    if not is_window_open(window_name):
        return
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def wait_for_preview_close(window_name: str) -> None:
    old_terminal_settings = None
    terminal_is_tty = sys.stdin.isatty()

    if sys.platform != "win32" and terminal_is_tty:
        old_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    try:
        while is_window_open(window_name):
            if cv2.waitKey(50) != -1:
                break

            if sys.platform == "win32":
                if msvcrt.kbhit():
                    msvcrt.getch()
                    break
            elif terminal_is_tty and select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
                break
    finally:
        if old_terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_settings)
        close_window_if_open(window_name)


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

def accept_identification(score: float, confidence: float) -> bool:
    if confidence >= 0.90:
        return score >= 0.60

    if confidence >= 0.75:
        return score >= ObjectIdentifier.MIN_SCORE

    return False

def box_xyxy(box) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    return x1, y1, x2, y2

def box_midpoint(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box_xyxy(box)
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def box_area(box) -> float:
    x1, y1, x2, y2 = box_xyxy(box)
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_contains_point(px, py, box) -> bool:
    x1, y1, x2, y2 = box_xyxy(box)
    return x1 <= px <= x2 and y1 <= py <= y2

def point_to_box_distance(px, py, box):
    x1, y1, x2, y2 = box_xyxy(box)

    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)

    return (dx * dx + dy * dy) ** 0.5

def select_touched_box(touch_point, boxes, max_distance_px):
    px, py = touch_point

    containing_boxes = [box for box in boxes if box_contains_point(px, py, box)]
    if containing_boxes:
        return min(containing_boxes, key=box_area)

    best_box = None
    best_distance = float("inf")

    for box in boxes:
        distance = point_to_box_distance(px, py, box)

        if distance < best_distance:
            best_distance = distance
            best_box = box

    if best_box is None:
        return None

    if best_distance > max_distance_px:
        return None

    return best_box

def execute_pipeline(frame, object_detector, hand_detector, object_identifier, timestamp_ms=0, verbose=True):
    # Run object detection
    boxes, _ = object_detector.detect_from_frame(frame, verbose=verbose)
    boxes = sorted(boxes, key=box_midpoint)

    # Convert boxes to ProductDetection objects
    detections: list[ProductDetection] = []
    for box in boxes:
        x1, y1, x2, y2 = box_xyxy(box)
        confidence = float(box.conf.item())
        detections.append(ProductDetection(bbox=(x1, y1, x2, y2), confidence=confidence))

    # Run hand detection
    hand_detection = hand_detector.detect_from_frame(frame, timestamp_ms=timestamp_ms, verbose=verbose)
 
    #touched_box = select_touched_box(touch_point, boxes, max_distance_px=50) if found else None

    identifications = object_identifier.identify_boxes(frame, detections, verbose=verbose)

    """
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box_xyxy(box)

        # Convert to int
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Optional but recommended: clamp coordinates to image size
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            if verbose:
                print(f"Skipping empty crop for box {i}: {(x1, y1, x2, y2)}")
            continue

        cropped_images.append(crop)

        # send cropped image to object identifier
        result_identification = object_identifier.identify_product(crop, k=5, verbose=verbose)

        product = result_identification["product"]

        description = product.get("description", "unknown")
        category = product.get("category", "unknown")

        #color = (0, 255, 0) if accept_identification(result_identification["score"], result_identification["confidence"]) else (0, 0, 255)
        color = (0, 255, 0) if result_identification["accepted"] else (0, 0, 255)
        if touched_box is not None and box is touched_box:
            color = (255, 0, 0)
        annotate_box(preview_image, boxes, i, color)
        #if not accept_identification(result_identification["score"], result_identification["confidence"]):
        if not result_identification["accepted"]:
            if verbose:
                print(f"[{i}] Unknown product, best: {description} ({category}), Score {result_identification['score']:.4f}, Confidence {result_identification['confidence']:.4f}")
            continue

        if verbose:
            print(f"[{i}] {description} ({category}), Score {result_identification['score']:.4f}, Confidence {result_identification['confidence']:.4f}")
    """
    return detections, hand_detection, identifications


def process_image(image_path: str, save: bool = False) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Cannot open image file: {image_path}")

    detector = ObjectDetector()
    hand_detector = HandDetector(mode=HandDetector.Mode.IMAGE)
    object_identifier = ObjectIdentifier()

    detections, hand_detection, identifications = execute_pipeline(frame, detector, hand_detector, object_identifier)


    scene_memory.update(
        0, 
        detections, 
        identifications, 
        None if not hand_detection.found else hand_detection.touched_point)    
 
    annotated_frame = scene_memory.annotate_image(frame)

    if save:
        # Save annotated image
        output_path = Path(f"{Path(image_path).stem}_pred.jpg")
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"Saved annotated image to: {output_path}")

    window_name = "Product Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, annotated_frame)
    print("Press any key to close the preview window.")
    wait_for_preview_close(window_name)

    return 0


def convert_video_for_mobile_sharing(input_path: Path, output_path: Path) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print("ffmpeg not found; saved the OpenCV MP4 without WhatsApp compatibility conversion.")
        return False

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-tag:v",
        "avc1",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        print("ffmpeg could not create a WhatsApp-compatible MP4.")
        if exc.stderr:
            print(exc.stderr.strip())
        return False

    return True


def process_video(video_path: str, save: bool = False) -> int:
    # Open video file
    cap = cv2.VideoCapture(filename=video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    detector = ObjectDetector()
    hand_detector = HandDetector(mode=HandDetector.Mode.VIDEO)
    object_identifier = ObjectIdentifier()

   
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define video writer
    out = None
    if save:
        output_path = Path(f"{Path(video_path).stem}_pred.mp4")
        raw_output_path = Path(f"{Path(video_path).stem}_pred_opencv.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Cannot create output video file: {raw_output_path}")
    
    print(f"Video: {Path(video_path).name}")
    print(f"FPS: {fps}, Resolution: {width}x{height}, Total frames: {total_frames}")
    print("Processing video...")
    print("Press 'q' to close the preview window.")

    window_name = "Annotated Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
      
        frame_count += 1
        
        # Run inference on frame
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        preview_image, cropped_images, detections = execute_pipeline(
            frame,
            detector,
            hand_detector,
            object_identifier,
            timestamp_ms,
        )

        # Display annotated frame
        cv2.imshow(window_name, preview_image)
        
        # Process GUI events so the window updates and can receive key presses
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview stopped by user.")
            break
        if not is_window_open(window_name):
            print("Preview window closed by user.")
            break
        
        # Write frame to output video
        if save and out:
            out.write(preview_image)
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    if out:
        out.release()
        if convert_video_for_mobile_sharing(raw_output_path, output_path):
            raw_output_path.unlink(missing_ok=True)
            print(f"Saved WhatsApp-compatible annotated video to: {output_path}")
        else:
            raw_output_path.replace(output_path)
            print(f"Saved annotated video to: {output_path}")
    close_window_if_open(window_name)

    return 0

def start_webcam(
    webcam_index: int,
    save: bool = False,
    stop_event: threading.Event | None = None,
) -> int:
    """Run webcam processing while writing its output to a RAM-backed log file."""
    log_directory = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path(tempfile.gettempdir())
    log_path = log_directory / "vista-webcam.log"

    print(f"Webcam output is being written to: {log_path}")
    print(f"Follow it from another terminal with: tail -f {log_path}")

    stdout_router, stderr_router = install_thread_output_routers()
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        with stdout_router.redirect(log_file), stderr_router.redirect(log_file):
            return process_webcam(webcam_index, save=save, stop_event=stop_event)


def open_webcam_capture(webcam_index: int) -> cv2.VideoCapture:
    """Open a camera using the native backend, with OpenCV fallback."""
    # V4L2 is Linux-specific. macOS cameras are exposed through AVFoundation.
    if sys.platform.startswith("linux"):
        preferred_backend = cv2.CAP_V4L2
    elif sys.platform == "darwin":
        preferred_backend = cv2.CAP_AVFOUNDATION
    else:
        preferred_backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(webcam_index, preferred_backend)
    if not cap.isOpened() and preferred_backend != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(webcam_index, cv2.CAP_ANY)

    if not cap.isOpened():
        if sys.platform == "darwin":
            raise ValueError(
                f"Cannot open webcam: {webcam_index}. Check that macOS has granted "
                "Camera permission to the app that launches Python (for example, "
                "Terminal or your IDE), then close any other app using the camera."
            )
        raise ValueError(f"Cannot open webcam: {webcam_index}")

    return cap


def print_webcam_info(webcam_index: int) -> None:
    """Print camera details and the common modes accepted by its OpenCV backend."""
    cap = open_webcam_capture(webcam_index)
    try:
        backend = cap.getBackendName() if hasattr(cap, "getBackendName") else "unknown"
        fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * offset)) & 0xFF) for offset in range(4)).rstrip("\x00")

        print(f"Webcam {webcam_index}")
        print(f"Backend: {backend}")
        print(
            "Current mode: "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"at {cap.get(cv2.CAP_PROP_FPS):g} FPS"
        )
        print(f"Pixel format: {fourcc or 'not reported'}")
        print(f"Reported buffer size: {cap.get(cv2.CAP_PROP_BUFFERSIZE):g}")

        # OpenCV does not expose a portable API for enumerating every mode. Probe
        # practical modes and report the dimensions/FPS actually selected by the
        # device, rather than assuming that every requested value is supported.
        requested_modes = (
            (640, 480, 30), (640, 480, 60),
            (800, 600, 30),
            (1280, 720, 30), (1280, 720, 60),
            (1280, 960, 30),
            (1024, 768, 10), (1024, 768, 15), (1024, 768, 25),
            (1024, 768, 30), (1024, 768, 60),
            (1920, 1080, 30), (1920, 1080, 60),
            (2560, 1440, 30), (3840, 2160, 30),
        )
        accepted_modes: dict[tuple[int, int, float], list[str]] = {}
        for requested_width, requested_height, requested_fps in requested_modes:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
            cap.set(cv2.CAP_PROP_FPS, requested_fps)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            mode = (actual_width, actual_height, actual_fps)
            accepted_modes.setdefault(mode, []).append(
                f"{requested_width}x{requested_height}@{requested_fps}"
            )

        print("Common modes accepted by the camera:")
        for (width, height, fps), requests in accepted_modes.items():
            requested = ", ".join(requests)
            print(f"  {width}x{height} at {fps:g} FPS (for request: {requested})")
    finally:
        cap.release()


def process_webcam(
    webcam_index: int,
    save: bool = False,
    stop_event: threading.Event | None = None,
) -> int:
    cap = open_webcam_capture(webcam_index)

    video_config = {
        0: {"width": 640, "height": 480, "fps": 30},
        1: {"width": 800, "height": 600, "fps": 25},
        2: {"width": 1024, "height": 768, "fps": 10},
        3: {"width": 1280, "height": 720, "fps": 30},
    }

    video_config_index = 2

    # YUYV is a V4L2 pixel format and is not meaningful for AVFoundation.
    if sys.platform.startswith("linux"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config[video_config_index]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config[video_config_index]["height"])
    cap.set(cv2.CAP_PROP_FPS, video_config[video_config_index]["fps"])

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"FPS: {fps}, Resolution: {width}x{height}")
    
    # Define video writer
    out = None
    output_path = Path("webcam_output.mp4")
    raw_output_path = Path("webcam_output_raw.mp4")
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise ValueError(f"Cannot create output video file: {raw_output_path}")

    
    print("Processing webcam...")
    print("Press 'q' to close the preview window.")

    window_name = "Annotated Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create the processing pipeline objects
    detector = ObjectDetector()
    hand_detector = HandDetector(mode=HandDetector.Mode.VIDEO)
    object_identifier = ObjectIdentifier()

    scene_memory.reset()

    frame_count = 0
    # Live-camera backends do not guarantee CAP_PROP_POS_MSEC: AVFoundation in
    # particular may repeatedly return zero. MediaPipe's video-mode hand
    # detector requires strictly increasing timestamps, so use a monotonic
    # local clock and enforce a one-millisecond minimum step between frames.
    webcam_started_at = time.monotonic()
    last_timestamp_ms = -1
    
    while stop_event is None or not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        if stop_event is not None and stop_event.is_set():
            break
      
        frame_count += 1
        
        # Run inference on frame
        timestamp_ms = int((time.monotonic() - webcam_started_at) * 1000)
        timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
        last_timestamp_ms = timestamp_ms
        # Measure elapsed time for inference
        start_time = time.time()

        verbose = frame_count % 30 == 0

        print("=" * 80)

        detections, hand_detection, identifications = execute_pipeline(
            frame,
            detector,
            hand_detector,
            object_identifier,
            timestamp_ms,
            verbose=False,
        )
        end_time = time.time()
        inference_time = end_time - start_time
        # Print inference time every 30 frames
        if frame_count % 30 == 0:
            print(f"Inference speed: {1.0/inference_time:.3f} frames/s")

        scene_memory.update(
            frame_count, 
            detections, 
            identifications, 
            None if not hand_detection.found else hand_detection.touched_point)

        # Display annotated frame
        # if hand_detection.annotated_image is not None:
        #     preview_image = hand_detection.annotated_image.copy()
        # else:
        #     preview_image = frame.copy()
        annotated_frame = scene_memory.annotate_image(frame)
        cv2.imshow(window_name, annotated_frame)
        
        # Process GUI events so the window updates and can receive key presses
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview stopped by user.")
            break
        if not is_window_open(window_name):
            print("Preview window closed by user.")
            break
        
        # Write frame to output video
        if save and out:
            out.write(annotated_frame)

    if stop_event is not None and stop_event.is_set():
        print("Webcam stop requested.")

    # Release resources
    cap.release()
    if out:
        out.release()
        if convert_video_for_mobile_sharing(raw_output_path, output_path):
            raw_output_path.unlink(missing_ok=True)
            print(f"Saved WhatsApp-compatible annotated video to: {output_path}")
        else:
            raw_output_path.replace(output_path)
            print(f"Saved annotated video to: {output_path}")
    close_window_if_open(window_name)

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
        type=int,
        default=None,
        metavar="DEVICE",
        help="Video device index to open for real-time inference.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated image to disk.",
    )

    return parser.parse_args()

class CommandInterpreter(cmd.Cmd):
    intro = "Interactive CLI. Type help or ? to list commands."
    prompt = "vista> "

    def __init__(self):
        super().__init__()
        self._webcam_jobs: queue.Queue[tuple[int, threading.Event] | None] = queue.Queue()
        self._webcam_lock = threading.Condition()
        self.webcam_stop_event: threading.Event | None = None
        self.webcam_running = False
        self.webcam_thread = threading.Thread(
            target=self._webcam_worker,
            name="webcam-processing",
        )
        self.webcam_thread.start()

    def _webcam_worker(self) -> None:
        """Process every webcam session on one long-lived thread for Qt/OpenCV."""
        while True:
            job = self._webcam_jobs.get()
            if job is None:
                return

            device_index, stop_event = job
            try:
                start_webcam(device_index, stop_event=stop_event)
            except Exception as exc:
                print(f"Webcam processing failed: {exc}")
            finally:
                with self._webcam_lock:
                    if self.webcam_stop_event is stop_event:
                        self.webcam_stop_event = None
                        self.webcam_running = False
                    self._webcam_lock.notify_all()

    def do_process_image(self, arg: str) -> None:
        """process_image <image_path>"""
        parts = shlex.split(arg)

        if len(parts) != 1:
            print("Usage: process_image <image_path>")
            return

        # If path is relative, resolve it against the current working directory
        image_path = Path(parts[0])
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path

        if not image_path.exists():
            print(f"Error: image file does not exist: {image_path}")
            return

        print(f"Processing image: {image_path}")
        image_thread = threading.Thread(
            target=process_image,
            args=(str(image_path),),
            name="image-processing",
        )
        image_thread.start()

    def do_process_video(self, arg: str) -> None:
            """process_video <video_path>"""
            parts = shlex.split(arg)

            if len(parts) != 1:
                print("Usage: process_video <video_path>")
                return

            video_path = Path(parts[0])

            if not video_path.exists():
                print(f"Error: video file does not exist: {video_path}")
                return

            process_video(video_path.name)

    def do_start_webcam(self, arg: str) -> None:
        """start_webcam [device_index] -- start webcam processing."""
        parts = shlex.split(arg)

        if len(parts) > 1:
            print("Usage: start_webcam [device_index]")
            return

        device_index = 0
        if parts:
            try:
                device_index = int(parts[0])
            except ValueError:
                print("Error: device_index must be an integer, for example: start_webcam 0")
                return

        # OpenCV's macOS GUI backend (Cocoa) must create and update windows from
        # the application's main thread. Running it in the background worker
        # results in the unhelpful "Unknown C++ exception" message. The command
        # remains interactive on macOS: press q in the preview to return here.
        if sys.platform == "darwin":
            print("Starting webcam on the main thread. Press 'q' in the preview to stop.")
            try:
                process_webcam(device_index)
            except Exception as exc:
                print(f"Webcam processing failed: {exc}")
            return

        with self._webcam_lock:
            if self.webcam_running:
                print("Webcam processing is already running. Use stop_webcam before starting another one.")
                return

            self.webcam_stop_event = threading.Event()
            self.webcam_running = True
            self._webcam_jobs.put((device_index, self.webcam_stop_event))

        print(f"Started webcam processing on device {device_index}. Use stop_webcam to stop it.")

    def do_webcam_info(self, arg: str) -> None:
        """webcam_info <device_index> -- show camera details and common supported modes."""
        parts = shlex.split(arg)
        if len(parts) != 1:
            print("Usage: webcam_info <device_index>")
            return

        try:
            device_index = int(parts[0])
        except ValueError:
            print("Error: device_index must be an integer, for example: webcam_info 0")
            return

        with self._webcam_lock:
            if self.webcam_running:
                print("Stop webcam processing before requesting webcam information.")
                return

        try:
            print_webcam_info(device_index)
        except Exception as exc:
            print(f"Could not read webcam information: {exc}")

    def do_stop_webcam(self, arg: str) -> None:
        """stop_webcam -- request webcam processing to stop."""
        if arg.strip():
            print("Usage: stop_webcam")
            return

        with self._webcam_lock:
            if self.webcam_stop_event is None:
                print("Webcam processing is not running.")
                return

            self.webcam_stop_event.set()
            if self._webcam_lock.wait_for(lambda: not self.webcam_running, timeout=5):
                print("Webcam processing stopped.")
            else:
                print("Webcam stop requested; waiting for the current frame to finish.")

    def do_exit(self, arg: str) -> bool:
        """exit"""
        if self.webcam_stop_event is not None:
            print("Stopping webcam processing...")
            self.webcam_stop_event.set()
            with self._webcam_lock:
                self._webcam_lock.wait_for(lambda: not self.webcam_running, timeout=5)
        self._webcam_jobs.put(None)
        self.webcam_thread.join(timeout=5)
        print("Exiting.")
        return True

    def do_quit(self, arg: str) -> bool:
        """quit"""
        return self.do_exit(arg)

    def do_EOF(self, arg: str) -> bool:
        """Exit when Ctrl+D sends an end-of-file signal."""
        print()
        return self.do_exit(arg)

    def emptyline(self) -> None:
        # Prevent repeating the previous command when the user presses Enter.
        pass

    def default(self, line: str) -> None:
        print(f"Unknown command: {line}")
        print("Type help or ? to list available commands.")

    def speak(self, text: str) -> None:
        """Speak the given text using the system's text-to-speech engine."""
        # For the moment, simply print the text.
        print(text)

    def do_listen(self, arg: str) -> None:
        """listen -- start listening for voice commands."""
        print("Listening for voice commands for 5 seconds...")
        detected_text = voice_processor.process_voice_command()
        print(f"Detected voice command: {detected_text}")

        intent, target, confidence = intent_classifier.classify(detected_text)
        print(f"Intent: {intent}, Target: {target}, Confidence: {confidence:.2f}")

        if intent == Intent.DESCRIBE_SCENE:
            self.speak(scene_memory.describe_scene())
        elif intent == Intent.DESCRIBE_POINTED_PRODUCT:
            self.speak(scene_memory.describe_pointed_product())
        elif intent == Intent.NAVIGATE_TO_TARGET:
            if target is None:
                self.speak("No he entendido el producto que estás buscando. ¿Puedes repetir?")
            else:
                self.speak(f"[SceneMemory] Navigating to target: {target}")
        elif intent == Intent.UNKNOWN:
            self.speak("No te he entendido. ¿Puedes repetir?")

    def do_describe_scene(self, arg: str) -> None:
        """Describe the current scene."""
        description = scene_memory.describe_scene()
        self.speak(description)

    def do_describe_pointed_product(self, arg: str) -> None:
        """Describe the product that is currently being pointed at."""
        description = scene_memory.describe_pointed_product()
        self.speak(description)

def main():

    args = parse_args()

    counter = sum([bool(args.image), bool(args.video), args.webcam is not None])
    if counter > 1:
        print("Error: Please provide only one of --image, --video, or --webcam.")
        exit(1)
    
    if counter == 0:
        CommandInterpreter().cmdloop()
        exit(0)

    if args.image:
        exit(process_image(args.image, save=args.save))

    if args.video:
        exit(process_video(args.video, save=args.save))

    if args.webcam is not None:
        exit(process_webcam(args.webcam, save=args.save))

if __name__ == "__main__":
    main()
