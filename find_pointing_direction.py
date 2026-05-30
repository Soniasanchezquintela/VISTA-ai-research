import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

# Get it manually here:
# wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

"""
# 1. Automatic Model Download Configuration (Production URL)
MODEL_PATH = "hand_landmarker.task"
# Google's active production CDN endpoint for the Tasks API bundle
MODEL_URL = "https://googleapis.com"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model file...")
    # Inject standard headers to satisfy Ubuntu 24.04 security lookups
    req = urllib.request.Request(
        MODEL_URL, 
        headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64)'}
    )
    with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
        out_file.write(response.read())
    print("Download completed successfully!")
"""

# Base MediaPipe Tasks aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 1. Define configuration options
# NOTE: Download the 'hand_landmarker.task' file and place it in the same folder!
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# Initialize the detector
detector = HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    height, width, _ = frame.shape

    # Flip horizontally for natural mirror view, convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert OpenCV frame to MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Get current timestamp in milliseconds
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    
    # Run hand landmark detection
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if detection_result.hand_landmarks:
        # Get landmarks for the first detected hand
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # In MediaPipe Tasks, landmarks are indices: 
        # Index 5 = INDEX_FINGER_MCP (Knuckle base)
        # Index 8 = INDEX_FINGER_TIP
        base_lm = hand_landmarks[5]
        tip_lm = hand_landmarks[8]

        # Calculate 3D direction vector (Tip - Base)
        vector = np.array([tip_lm.x - base_lm.x, tip_lm.y - base_lm.y, tip_lm.z - base_lm.z])
        magnitude = np.linalg.norm(vector)
        
        if magnitude > 0:
            direction = vector / magnitude
        else:
            direction = np.array([0, 0, 0])

        # Convert normalized tip coordinates to pixel coordinates
        start_point = (int(tip_lm.x * width), int(tip_lm.y * height))

        # Project the 2D direction components to calculate the arrow endpoint
        arrow_length = 100
        dx, dy = direction[0], direction[1]
        
        end_point = (
            int(start_point[0] + dx * arrow_length),
            int(start_point[1] + dy * arrow_length)
        )

        # Draw the arrowed line on the frame
        cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 3, tipLength=0.3)

        # Display raw vector numbers on screen for debugging
        cv2.putText(frame, f"Dir: [{dx:.2f}, {dy:.2f}]", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Pointing Arrow Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
