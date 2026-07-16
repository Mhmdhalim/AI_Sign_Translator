from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "word_classifier.onnx"
CLASSES_PATH = PROJECT_ROOT / "models" / "word_classes.npy"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 258
CONFIDENCE_THRESHOLD = 0.50
SMOOTHING_WINDOW = 5
PREDICTION_COOLDOWN = 1.5


def extract_pose_features(pose_landmarks):
    if pose_landmarks is None:
        return np.zeros(132, dtype=np.float32)

    features = []

    for landmark in pose_landmarks.landmark:
        features.extend(
            [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility,
            ]
        )

    return np.asarray(features, dtype=np.float32)


def extract_hand_features(hand_landmarks):
    if hand_landmarks is None:
        return np.zeros(63, dtype=np.float32)

    features = []

    for landmark in hand_landmarks.landmark:
        features.extend(
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ]
        )

    return np.asarray(features, dtype=np.float32)


def extract_frame_features(results):
    pose_features = extract_pose_features(results.pose_landmarks)
    left_hand_features = extract_hand_features(
        results.left_hand_landmarks
    )
    right_hand_features = extract_hand_features(
        results.right_hand_landmarks
    )

    features = np.concatenate(
        [
            pose_features,
            left_hand_features,
            right_hand_features,
        ]
    )

    if features.shape != (FEATURE_SIZE,):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    return features


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Classes not found: {CLASSES_PATH}")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    classes = np.load(CLASSES_PATH, allow_pickle=True)
    input_name = session.get_inputs()[0].name

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)

    current_prediction = ""
    confidence = 0.0
    last_accepted_word = ""
    last_prediction_time = 0.0

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    print("Real-time word recognition started.")
    print("Hold the complete sign inside the camera.")
    print("R: reset sequence")
    print("Q: quit")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(rgb_frame)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
            )

            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
            )

            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
            )

            frame_features = extract_frame_features(results)
            sequence.append(frame_features)

            if len(sequence) == SEQUENCE_LENGTH:
                input_sequence = np.expand_dims(
                    np.asarray(sequence, dtype=np.float32),
                    axis=0,
                )

                probabilities = session.run(
                    None,
                    {
                        input_name: input_sequence.astype(np.float32)
                    },
                )[0][0]

                class_index = int(np.argmax(probabilities))
                predicted_word = str(classes[class_index])
                confidence = float(probabilities[class_index])

                prediction_history.append(predicted_word)

                if (
                    len(prediction_history) == SMOOTHING_WINDOW
                    and len(set(prediction_history)) == 1
                    and confidence >= CONFIDENCE_THRESHOLD
                ):
                    now = time.time()

                    if (
                        predicted_word != last_accepted_word
                        or now - last_prediction_time
                        >= PREDICTION_COOLDOWN
                    ):
                        current_prediction = predicted_word
                        last_accepted_word = predicted_word
                        last_prediction_time = now

            cv2.putText(
                frame,
                f"Prediction: {current_prediction or '-'}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "R: Reset | Q: Quit",
                (30, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            cv2.imshow("ASL Word Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                sequence.clear()
                prediction_history.clear()
                current_prediction = ""
                confidence = 0.0
                last_accepted_word = ""

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
# python -c "import numpy as np; print(np.load('models/word_classes.npy', allow_pickle=True))"
# ['before' 'bowling' 'brother' 'candy' 'carrot' 'champion' 'change'
#  'computer' 'cool' 'cousin' 'deaf' 'family' 'hot' 'tall' 'thanksgiving'
#  'thin' 'trade' 'what' 'who' 'yes']
