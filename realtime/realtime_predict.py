import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load model + classes
model = load_model("models/sign_classifier.keras")
classes = np.load("models/classes.npy")

# Text building
sentence = ""
current_word = ""
last_letter = ""
stable_count = 0
STABILITY_THRESHOLD = 12

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

print("Running real-time prediction...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    letter = ""

    # =========================
    # HAND DETECTION + PREDICTION
    # =========================
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data = np.array(data).reshape(1, 63)

            pred = model.predict(data, verbose=0)
            class_id = np.argmax(pred)
            letter = classes[class_id]

    # =========================
    # WORD BUILDER LOGIC (FIXED)
    # =========================
    if letter != "":
        if letter == last_letter:
            stable_count += 1
        else:
            stable_count = 0
            last_letter = letter

        if stable_count == STABILITY_THRESHOLD:
            if letter != "Nothing":
                current_word += letter

            stable_count = 0

    # =========================
    # DISPLAY TEXT
    # =========================
    cv2.putText(frame, f"Word: {current_word}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Sentence: {sentence}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("ASL Real-Time", frame)

    # =========================
    # KEY CONTROLS
    # =========================
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    elif key == ord(' '):
        sentence += current_word + " "
        current_word = ""

    elif key == ord('r'):
        sentence = ""
        current_word = ""

cap.release()
cv2.destroyAllWindows()