#! /usr/bin/env python3
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field
import argparse
import cmd
import cv2
import logging
import queue
import shlex
import shutil
import subprocess
import sys
import time
import threading

print("Loading required modules...", flush=True)

from object_detector import ObjectDetector
from hand_detector import HandDetector
from object_identifier import ObjectIdentifier
from scene_memory import ProductIdentification
from scene_memory.tracked_scene_memory import TrackedShelfSceneMemory as ShelfSceneMemory
from voice_to_text import VoiceCommandProcessor
from intent_classifier import SimpleIntentClassifier, Intent


# Global objects for the interactive CLI and scene processing
MAIN_THREAD_LOG_PATH = Path("/tmp/vista-main.log")
logger = logging.getLogger("vista")
scene_memory = ShelfSceneMemory()
scene_memory_lock = threading.Lock()
voice_processor = VoiceCommandProcessor()
intent_classifier = SimpleIntentClassifier()


@dataclass
class MainThreadCommand:
    name: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    reply_queue: queue.Queue | None = None


class ThreadOutputRouter:
    """Route stdout/stderr per thread while leaving other threads untouched."""

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


def configure_logging() -> None:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return

    handler = logging.FileHandler(MAIN_THREAD_LOG_PATH, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


def install_thread_output_routers() -> tuple[ThreadOutputRouter, ThreadOutputRouter]:
    global _stdout_router, _stderr_router

    if _stdout_router is None:
        _stdout_router = ThreadOutputRouter(sys.stdout)
        sys.stdout = _stdout_router
    if _stderr_router is None:
        _stderr_router = ThreadOutputRouter(sys.stderr)
        sys.stderr = _stderr_router

    return _stdout_router, _stderr_router


@contextmanager
def redirect_main_thread_output():
    stdout_router, stderr_router = install_thread_output_routers()
    with MAIN_THREAD_LOG_PATH.open("a", encoding="utf-8", buffering=1) as log_file:
        with stdout_router.redirect(log_file), stderr_router.redirect(log_file):
            yield

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
    try:
        while is_window_open(window_name):
            if cv2.waitKey(50) != -1:
                break
    finally:
        close_window_if_open(window_name)

def execute_pipeline(frame, object_detector, hand_detector, object_identifier, timestamp_ms=0, verbose=True):
    # Run object detection
    detections, _ = object_detector.detect_from_frame(frame, verbose=verbose)

    # Run hand detection
    hand_detection = hand_detector.detect_from_frame(frame, timestamp_ms=timestamp_ms, verbose=verbose)

    # Identify products in the detected boxes
    identifications = object_identifier.identify_boxes(frame, detections, verbose=verbose)

    return detections, hand_detection, identifications

def process_image(image_path: str, save: bool = False) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Cannot open image file: {image_path}")

    detector = ObjectDetector()
    hand_detector = HandDetector(mode=HandDetector.Mode.IMAGE)
    object_identifier = ObjectIdentifier()

    detections, hand_detection, identifications = execute_pipeline(frame, detector, hand_detector, object_identifier)


    with scene_memory_lock:
        scene_memory.update(
            0,
            detections,
            identifications,
            hand_detection)

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
        detections, hand_detection, identifications = execute_pipeline(
            frame,
            detector,
            hand_detector,
            object_identifier,
            timestamp_ms,
            verbose=False,
        )

        with scene_memory_lock:
            scene_memory.update(
                frame_count,
                detections,
                identifications,
                hand_detection)

            annotated_frame = scene_memory.annotate_image(frame)

        # Display annotated frame
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


class WebcamSession:
    def __init__(self, webcam_index: int, save: bool = False):
        self.cap = open_webcam_capture(webcam_index)
        self.save = save
        self.out = None
        self.output_path = Path("webcam_output.mp4")
        self.raw_output_path = Path("webcam_output_raw.mp4")
        self.window_name = "Annotated Video"
        self.frame_count = 0
        self.webcam_started_at = time.monotonic()
        self.last_timestamp_ms = -1
        self.closed = False

        video_config = {
            0: {"width": 640, "height": 480, "fps": 30},
            1: {"width": 800, "height": 600, "fps": 25},
            2: {"width": 1024, "height": 768, "fps": 10},
            3: {"width": 1280, "height": 720, "fps": 30},
        }
        video_config_index = 2

        # YUYV is a V4L2 pixel format and is not meaningful for AVFoundation.
        if sys.platform.startswith("linux"):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config[video_config_index]["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config[video_config_index]["height"])
        self.cap.set(cv2.CAP_PROP_FPS, video_config[video_config_index]["fps"])

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"FPS: {self.fps}, Resolution: {self.width}x{self.height}")

        if self.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(
                str(self.raw_output_path),
                fourcc,
                self.fps,
                (self.width, self.height),
            )
            if not self.out.isOpened():
                self.cap.release()
                raise ValueError(f"Cannot create output video file: {self.raw_output_path}")

        print("Processing webcam...")
        print("Press 'q' to close the preview window, or type stop_webcam.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.detector = ObjectDetector()
        self.hand_detector = HandDetector(mode=HandDetector.Mode.VIDEO)
        self.object_identifier = ObjectIdentifier()

        with scene_memory_lock:
            scene_memory.reset()

    def step(self) -> bool:
        ret, frame = self.cap.read()
        if not ret:
            return False

        self.frame_count += 1
        timestamp_ms = int((time.monotonic() - self.webcam_started_at) * 1000)
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        start_time = time.time()
        verbose = self.frame_count % 15 == 0

        if verbose:
            print("=" * 80)

        detections, hand_detection, identifications = execute_pipeline(
            frame,
            self.detector,
            self.hand_detector,
            self.object_identifier,
            timestamp_ms,
            verbose=False,
        )
        inference_time = time.time() - start_time

        if verbose:
            print(f"Inference speed: {1.0/inference_time:.3f} frames/s")

        with scene_memory_lock:
            scene_memory.update(
                self.frame_count,
                detections,
                identifications,
                hand_detection,
                verbose=verbose)
            annotated_frame = scene_memory.annotate_image(frame, verbose)

        cv2.imshow(self.window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview stopped by user.")
            return False
        if not is_window_open(self.window_name):
            print("Preview window closed by user.")
            return False

        if self.save and self.out:
            self.out.write(annotated_frame)

        return True

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True
        self.cap.release()
        if self.out:
            self.out.release()
            if convert_video_for_mobile_sharing(self.raw_output_path, self.output_path):
                self.raw_output_path.unlink(missing_ok=True)
                print(f"Saved WhatsApp-compatible annotated video to: {self.output_path}")
            else:
                self.raw_output_path.replace(self.output_path)
                print(f"Saved annotated video to: {self.output_path}")
        close_window_if_open(self.window_name)


def process_webcam(
    webcam_index: int,
    save: bool = False,
    stop_event: threading.Event | None = None,
) -> int:
    session = WebcamSession(webcam_index, save=save)
    try:
        while stop_event is None or not stop_event.is_set():
            if not session.step():
                break

        if stop_event is not None and stop_event.is_set():
            print("Webcam stop requested.")
    finally:
        session.close()

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

    def __init__(self, main_command_queue: queue.Queue[MainThreadCommand]):
        super().__init__()
        self.main_command_queue = main_command_queue

    def _run_on_main_thread(self, name: str, *args, wait: bool = True, **kwargs):
        reply_queue = queue.Queue(maxsize=1) if wait else None
        self.main_command_queue.put(
            MainThreadCommand(
                name=name,
                args=args,
                kwargs=kwargs,
                reply_queue=reply_queue,
            )
        )

        if reply_queue is None:
            return True

        success, result = reply_queue.get()
        if not success:
            print(result)
            return False

        return True

    def _print_main_log_hint(self) -> None:
        print(f"Runtime output is being written to: {MAIN_THREAD_LOG_PATH}")

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

        self._print_main_log_hint()
        self._run_on_main_thread("process_image", str(image_path), False)

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

        self._print_main_log_hint()
        self._run_on_main_thread("process_video", str(video_path), False)

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

        print(f"Starting webcam on device {device_index}.")
        self._print_main_log_hint()
        self._run_on_main_thread("start_webcam", device_index, False)

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

        try:
            print_webcam_info(device_index)
        except Exception as exc:
            print(f"Could not read webcam information: {exc}")

    def do_stop_webcam(self, arg: str) -> None:
        """stop_webcam -- request webcam processing to stop."""
        if arg.strip():
            print("Usage: stop_webcam")
            return

        if self._run_on_main_thread("stop_webcam"):
            print("Webcam stopped.")

    def do_exit(self, arg: str) -> bool:
        """exit"""
        self._run_on_main_thread("exit", wait=False)
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
            with scene_memory_lock:
                description = scene_memory.describe_scene()
            self.speak(description)
        elif intent == Intent.DESCRIBE_POINTED_PRODUCT:
            with scene_memory_lock:
                description = scene_memory.describe_pointed_product()
            self.speak(description)
        elif intent == Intent.NAVIGATE_TO_TARGET:
            if target is None:
                self.speak("No he entendido el producto que estás buscando. ¿Puedes repetir?")
            else:
                self.speak(f"[SceneMemory] Navigating to target: {target}")
        elif intent == Intent.UNKNOWN:
            self.speak("No te he entendido. ¿Puedes repetir?")

    def do_describe_scene(self, arg: str) -> None:
        """Describe the current scene."""
        with scene_memory_lock:
            description = scene_memory.describe_scene()
        self.speak(description)

    def do_describe_pointed_product(self, arg: str) -> None:
        """Describe the product that is currently being pointed at."""
        with scene_memory_lock:
            description = scene_memory.describe_pointed_product()
        self.speak(description)

    def do_reset_scene_memory(self, arg: str) -> None:
        """Reset the scene memory."""
        with scene_memory_lock:
            scene_memory.reset()


def reply_to_command(command: MainThreadCommand, success: bool, result=None) -> None:
    if command.reply_queue is not None:
        command.reply_queue.put((success, result))


def handle_main_thread_command_with_redirect(
    command: MainThreadCommand,
    webcam_session: WebcamSession | None,
) -> tuple[WebcamSession | None, bool]:
    with redirect_main_thread_output():
        return handle_main_thread_command(command, webcam_session)


def handle_main_thread_command(
    command: MainThreadCommand,
    webcam_session: WebcamSession | None,
) -> tuple[WebcamSession | None, bool]:
    keep_running = True

    try:
        if command.name == "start_webcam":
            if webcam_session is not None:
                reply_to_command(command, False, "Webcam processing is already running.")
                return webcam_session, keep_running

            device_index, save = command.args
            logger.info("Starting webcam device %s", device_index)
            webcam_session = WebcamSession(device_index, save=save)
            reply_to_command(command, True, None)
            return webcam_session, keep_running

        if command.name == "stop_webcam":
            if webcam_session is None:
                reply_to_command(command, False, "Webcam processing is not running.")
                return webcam_session, keep_running

            logger.info("Webcam stop requested")
            print("Webcam stop requested.")
            webcam_session.close()
            reply_to_command(command, True, None)
            return None, keep_running

        if command.name == "process_image":
            if webcam_session is not None:
                reply_to_command(command, False, "Stop webcam processing before processing an image.")
                return webcam_session, keep_running

            image_path, save = command.args
            logger.info("Processing image %s", image_path)
            print(f"Processing image: {image_path}")
            result = process_image(image_path, save=save)
            reply_to_command(command, True, result)
            return webcam_session, keep_running

        if command.name == "process_video":
            if webcam_session is not None:
                reply_to_command(command, False, "Stop webcam processing before processing a video.")
                return webcam_session, keep_running

            video_path, save = command.args
            logger.info("Processing video %s", video_path)
            print(f"Processing video: {video_path}")
            result = process_video(video_path, save=save)
            reply_to_command(command, True, result)
            return webcam_session, keep_running

        if command.name in {"exit", "quit"}:
            if webcam_session is not None:
                logger.info("Stopping webcam processing during exit")
                print("Stopping webcam processing...")
                webcam_session.close()
            reply_to_command(command, True, None)
            return None, False

        reply_to_command(command, False, f"Unknown main-thread command: {command.name}")
        return webcam_session, keep_running
    except Exception as exc:
        logger.exception("%s failed", command.name)
        reply_to_command(command, False, f"{command.name} failed: {exc}")
        return webcam_session, keep_running


def run_main_thread_command_loop(command_queue: queue.Queue[MainThreadCommand]) -> None:
    webcam_session = None
    keep_running = True

    while keep_running:
        if webcam_session is None:
            command = command_queue.get()
            webcam_session, keep_running = handle_main_thread_command_with_redirect(command, webcam_session)
            continue

        try:
            command = command_queue.get_nowait()
        except queue.Empty:
            command = None

        if command is not None:
            webcam_session, keep_running = handle_main_thread_command_with_redirect(command, webcam_session)
            continue

        with redirect_main_thread_output():
            keep_webcam_running = webcam_session.step()

        if not keep_webcam_running:
            with redirect_main_thread_output():
                webcam_session.close()
                print("Webcam processing stopped.")
                logger.info("Webcam processing stopped")
            webcam_session = None

    if webcam_session is not None:
        with redirect_main_thread_output():
            webcam_session.close()


def main():

    args = parse_args()

    counter = sum([bool(args.image), bool(args.video), args.webcam is not None])
    if counter > 1:
        print("Error: Please provide only one of --image, --video, or --webcam.")
        exit(1)
    
    if counter == 0:
        configure_logging()
        install_thread_output_routers()
        main_command_queue: queue.Queue[MainThreadCommand] = queue.Queue()
        interpreter = CommandInterpreter(main_command_queue)
        command_thread = threading.Thread(
            target=interpreter.cmdloop,
            name="command-interpreter",
        )
        command_thread.start()
        run_main_thread_command_loop(main_command_queue)
        command_thread.join()
        exit(0)

    if args.image:
        exit(process_image(args.image, save=args.save))

    if args.video:
        exit(process_video(args.video, save=args.save))

    if args.webcam is not None:
        exit(process_webcam(args.webcam, save=args.save))

if __name__ == "__main__":
    main()
