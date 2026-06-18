import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import joblib

recording = False
current_label = None
predicted_letter = None
classifier = joblib.load("dataStorage/hand_sign_classifier.pkl")  # Load the trained classifier

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

# Set up the MediaPipe gesture recognizer
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model_path/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result, num_hands=2)

with HandLandmarker.create_from_options(options) as landmarker:
    latest_result = None
    file = open("dataStorage/landmarks.txt", "a")

    while True:
        key = cv2.waitKey(1)

        if key == ord('1'):
            recording = not recording
        elif key == ord('2'):
            break
        elif key != -1 and key < 128 and chr(key).isalpha():
            current_label = chr(key)

        timestamp = int(time.time() * 1000)
        ret, frame = cam.read()

        if not ret:
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)

        if latest_result:
            # Landmark drawer
            for hand_landmarks in latest_result.hand_landmarks:
                row = []  # flat list of x, y, x, y, ... for this frame
                # Draw circles for each landmark
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    row.append(landmark.x)
                    row.append(landmark.y)

                # Write to file only when recording
                if recording and current_label is not None:
                    file.write(",".join(str(v) for v in row) + "," + current_label + "\n")

                # Predict continuously every frame
                predicted_letter = classifier.predict([row])[0]

                # Draw lines between connected landmarks
                for start_index, end_index in HAND_CONNECTIONS:
                    start_landmark = hand_landmarks[start_index]
                    end_landmark = hand_landmarks[end_index]
                    x1 = int(start_landmark.x * frame.shape[1])
                    y1 = int(start_landmark.y * frame.shape[0])
                    x2 = int(end_landmark.x * frame.shape[1])
                    y2 = int(end_landmark.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display label, recording status, and prediction on screen
        cv2.putText(frame, f"Label: {current_label} | Recording: {recording}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Prediction: {predicted_letter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the captured frame
        cv2.imshow('Camera', frame)

    # Release the capture
    cam.release()
    file.close()