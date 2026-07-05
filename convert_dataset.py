import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

INPUT_PATH = "ASL_Dataset/Train"
OUTPUT_PATH = "landmark_dataset"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return []

    hand = results.multi_hand_landmarks[0]

    landmarks = []
    for lm in hand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return landmarks


for label in os.listdir(INPUT_PATH):

    label_path = os.path.join(INPUT_PATH, label)

    if not os.path.isdir(label_path):
        continue

    output_label_path = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_label_path, exist_ok=True)

    counter = 0

    for img_name in os.listdir(label_path):

        img_path = os.path.join(label_path, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        landmarks = extract_landmarks(image)

        if len(landmarks) == 0:
            continue

        save_path = os.path.join(output_label_path, f"{counter}.npy")

        np.save(save_path, np.array(landmarks))

        counter += 1

        print(f"Saved {save_path}")

print("Done")