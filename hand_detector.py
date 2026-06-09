import mediapipe as mp
import numpy as np
import cv2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # palm
]

def draw_hand_landmarks(image_bgr, hand_landmarks):
    height, width, _ = image_bgr.shape

    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            image_bgr,
            points[start_idx],
            points[end_idx],
            color=(0, 255, 0),
            thickness=2,
        )

    # Draw landmark points
    for x, y in points:
        cv2.circle(
            image_bgr,
            (x, y),
            radius=4,
            color=(0, 0, 255),
            thickness=-1,
        )


class HandDetector:

    class Mode:
        IMAGE = mp.tasks.vision.RunningMode.IMAGE
        VIDEO = mp.tasks.vision.RunningMode.VIDEO
        WEBCAM = mp.tasks.vision.RunningMode.LIVE_STREAM

    def __init__(self, mode=Mode.VIDEO):

        # Base MediaPipe Tasks aliases
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        #VisionRunningMode = mp.tasks.vision.RunningMode

        # 1. Define configuration options
        # NOTE: Download the 'hand_landmarker.task' file and place it in the same folder!
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=mode,
            num_hands=1
        )

        # Initialize the detector
        self.detector = HandLandmarker.create_from_options(options)
        self.mode = mode

    def __del__(self):
        if self.detector:
            self.detector.close()

    def get_touch_point(self, hand_landmarks, image_width, image_height):
        fingertip_ids = [8, 12, 16]  # index, middle, ring

        points = []
        for idx in fingertip_ids:
            lm = hand_landmarks[idx]
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)
            points.append((x, y))

        touch_x = int(sum(p[0] for p in points) / len(points))
        touch_y = int(sum(p[1] for p in points) / len(points))

        return touch_x, touch_y

    def detect_from_file(self, image_path: str):
        bgr_image = cv2.imread(image_path)

        if bgr_image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # Image mode does not need timestamps. For other modes, pass 0 as a safe default.
        return self.detect_from_frame(bgr_image)

    def detect_from_frame(self, frame, timestamp_ms: int = 0) -> tuple[bool, tuple[int, int], np.ndarray | None]:

        found = False
        touch_point = (0, 0)
        annotated_frame = None
        height, width, _ = frame.shape

        # Flip horizontally for natural mirror view, convert to RGB
        #frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert OpenCV frame to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Get current timestamp in milliseconds
        # timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Run hand landmark detection based on configured running mode.
        if self.mode == self.Mode.IMAGE:
            detection_result = self.detector.detect(mp_image)
        else:
            detection_result = self.detector.detect_for_video(mp_image, int(timestamp_ms))

        if detection_result.hand_landmarks:
            found = True
            # Get landmarks for the first detected hand
            hand_landmarks = detection_result.hand_landmarks[0]
    
            # Get touch point (average of index, middle, ring fingertips)
            touch_point = self.get_touch_point(hand_landmarks, width, height)

            annotated_frame = frame.copy()

            # Draw hand landmarks and connections without protobuf dependency.
            for hand_landmarks in detection_result.hand_landmarks:
                draw_hand_landmarks(annotated_frame, hand_landmarks)

            # Draw a blue circle at the touch point for visualization
            cv2.circle(annotated_frame, touch_point, radius=10, color=(255, 0, 0), thickness=-1)

            # Draw the arrowed line on the frame
            #cv2.arrowedLine(annotated_frame, start_point, end_point, (0, 255, 0), 3, tipLength=0.3)

            # Display raw vector numbers on screen for debugging
            #cv2.putText(annotated_frame, f"Dir: [{dx:.2f}, {dy:.2f}]", 
            #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            #print("No hand detected in this frame.")
            pass
        
        return found, touch_point, annotated_frame
    