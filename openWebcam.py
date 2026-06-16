import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Connections for drawing hand landmarks
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index finger
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17)               # palm
]

# Open the default camera
cam = cv2.VideoCapture(0)

# Initialize the MediaPipe gesture recognizer
model_path = 'model_path/hand_landmarker.task'

# Set up the MediaPipe gesture recognizer
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model_path/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result, num_hands=2)

with HandLandmarker.create_from_options(options) as landmarker:
    latest_result = None

    while True:
        timestamp = int(time.time() * 1000)
        ret, frame = cam.read()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)

        if latest_result:
            # Landmark drawer
            for hand_landmarks in latest_result.hand_landmarks:
                # Draw circles for each landmark
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw lines between connected landmarks
                for start_index, end_index in HAND_CONNECTIONS:
                    start_landmark = hand_landmarks[start_index]
                    end_landmark = hand_landmarks[end_index]
                    x1 = int(start_landmark.x * frame.shape[1])
                    y1 = int(start_landmark.y * frame.shape[0])
                    x2 = int(end_landmark.x * frame.shape[1])
                    y2 = int(end_landmark.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release the capture
    cam.release()