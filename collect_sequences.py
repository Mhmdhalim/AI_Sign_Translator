import cv2
import os
import numpy as np
import mediapipe as mp

from utils.recorder import Recorder

# ==========================
# Setup
# ==========================

DATA_PATH = "dataset"

os.makedirs(DATA_PATH, exist_ok=True)

# ==========================
# MediaPipe setup
# ==========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================
# Camera
# ==========================

cap = cv2.VideoCapture(0)

# ==========================
# Recorder
# ==========================

recorder = Recorder(sequence_length=60)

# ==========================
# Get Action Name
# ==========================

action = input("Enter action name (e.g. Hello, Thanks): ")
action_path = os.path.join(DATA_PATH, action)
os.makedirs(action_path, exist_ok=True)

counter = len(os.listdir(action_path))

print("\nPress 'R' to record sequence | Press 'Q' to quit\n")

# ==========================
# Main Loop
# ==========================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # Draw landmarks (preview only)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        f"Action: {action}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press R to Record",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Sign Language Dataset Creator", frame)

    key = cv2.waitKey(1) & 0xFF

    # ==========================
    # Start Recording
    # ==========================
    if key == ord("r"):

        save, video_path = recorder.record_sequence(
            cap,
            hands,
            mp_draw,
            mp_hands
        )

        if save:

            file_path = os.path.join(action_path, f"{counter}.npy")
            recorder.save(file_path, video_path)

            counter += 1

            print(f"[INFO] Saved sequence {counter}")

        else:
            print("[INFO] Discarded sequence")

    # Quit
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()