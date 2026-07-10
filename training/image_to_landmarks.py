import cv2
import os
import numpy as np
import mediapipe as mp

# ======================
# Setup MediaPipe
# ======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

DATASET_PATH = "asl_dataset"
OUTPUT_PATH = "landmark_dataset"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ======================
# Loop over folders
# ======================
for label in os.listdir(DATASET_PATH):

    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    output_label_path = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_label_path, exist_ok=True)

    for img_name in os.listdir(label_path):

        img_path = os.path.join(label_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        landmarks = []

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 0:
            continue

        # save
        file_name = img_name.replace(".jpg", "").replace(".png", "")
        save_path = os.path.join(output_label_path, file_name + ".npy")

        np.save(save_path, np.array(landmarks))

        print(f"Saved: {save_path}")