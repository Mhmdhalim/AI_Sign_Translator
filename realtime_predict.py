import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# تحميل الموديل
model = load_model("models/sign_classifier.keras")
classes = np.load("models/classes.npy")

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)


def extract_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    data = []
    for lm in hand.landmark:
        data.extend([lm.x, lm.y, lm.z])

    return np.array(data).reshape(1, -1)


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    landmarks = extract_landmarks(frame)

    if landmarks is not None:
        prediction = model.predict(landmarks, verbose=0)
        class_id = np.argmax(prediction)
        label = classes[class_id]

        cv2.putText(
            frame,
            str(label),
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3
        )

    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()