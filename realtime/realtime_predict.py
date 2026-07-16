import queue
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import pyttsx3
import pythoncom


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "sign_classifier.onnx"
CLASSES_PATH = PROJECT_ROOT / "models" / "classes.npy"

STABILITY_THRESHOLD = 6
RELEASE_THRESHOLD = 6
CONFIDENCE_THRESHOLD = 0.70
PREDICTION_INTERVAL = 2


def load_model_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(f"Classes file not found: {CLASSES_PATH}")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    classes = np.load(CLASSES_PATH, allow_pickle=True)
    input_name = session.get_inputs()[0].name

    return session, classes, input_name


def extract_landmarks(hand_landmarks):
    features = []

    for landmark in hand_landmarks.landmark:
        features.extend(
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ]
        )

    return np.asarray(features, dtype=np.float32).reshape(1, 63)


def choose_voice():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")

    if not voices:
        engine.stop()
        raise RuntimeError("No system voices were found.")

    print("\nAvailable voices:")

    for index, voice in enumerate(voices):
        voice_name = getattr(voice, "name", f"Voice {index}")
        print(f"{index}: {voice_name}")

    user_input = input(
        f"Choose voice number (0-{len(voices) - 1}): "
    ).strip()

    try:
        selected_index = int(user_input)
    except ValueError:
        selected_index = 0

    if selected_index < 0 or selected_index >= len(voices):
        selected_index = 0

    selected_voice_id = voices[selected_index].id
    selected_voice_name = voices[selected_index].name

    engine.stop()

    print(f"Selected voice: {selected_voice_name}")

    return selected_voice_id


def create_speech_system(voice_id):
    speech_queue = queue.Queue()

    def speech_worker():
        pythoncom.CoInitialize()

        try:
            engine = pyttsx3.init()
            engine.setProperty("voice", voice_id)
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)

            while True:
                text = speech_queue.get()

                try:
                    if text is None:
                        break

                    cleaned_text = str(text).strip()

                    if cleaned_text:
                        engine.stop()
                        engine.say(cleaned_text)
                        engine.runAndWait()

                except Exception as error:
                    print(f"Speech error: {error}")

                finally:
                    speech_queue.task_done()

            engine.stop()

        finally:
            pythoncom.CoUninitialize()

    speech_thread = threading.Thread(
        target=speech_worker,
        daemon=True,
    )
    speech_thread.start()

    return speech_queue, speech_thread


def reset_letter_state():
    return "", 0, 0, True


def main():
    session, classes, input_name = load_model_resources()

    voice_id = choose_voice()
    speech_queue, speech_thread = create_speech_system(voice_id)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        hands.close()
        speech_queue.put(None)
        raise RuntimeError("Could not open the webcam.")

    current_word = ""
    sentence = ""

    displayed_letter = ""
    last_detected_letter = ""

    stable_count = 0
    release_count = 0
    letter_locked = False

    confidence = 0.0
    frame_counter = 0

    print("\nReal-time ASL recognition started.")
    print("Click the camera window before using keyboard controls.")
    print("SPACE: add and speak the current word")
    print("ENTER: speak the full sentence")
    print("B: delete the last letter")
    print("R: reset all text")
    print("Q: quit")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)
            frame_counter += 1

            if results.multi_hand_landmarks:
                release_count = 0

                hand_landmarks = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                if frame_counter % PREDICTION_INTERVAL == 0:
                    features = extract_landmarks(hand_landmarks)

                    prediction = session.run(
                        None,
                        {input_name: features},
                    )[0][0]

                    class_index = int(np.argmax(prediction))
                    displayed_letter = str(classes[class_index])
                    confidence = float(prediction[class_index])

                    if confidence >= CONFIDENCE_THRESHOLD:
                        if displayed_letter == last_detected_letter:
                            stable_count += 1
                        else:
                            last_detected_letter = displayed_letter
                            stable_count = 1
                            letter_locked = False

                        if (
                            stable_count >= STABILITY_THRESHOLD
                            and not letter_locked
                            and displayed_letter.lower() != "nothing"
                        ):
                            current_word += displayed_letter
                            letter_locked = True
                            stable_count = 0
                    else:
                        stable_count = 0

            else:
                displayed_letter = ""
                confidence = 0.0

                release_count += 1
                stable_count = 0
                last_detected_letter = ""

                if release_count >= RELEASE_THRESHOLD:
                    letter_locked = False

            cv2.putText(
                frame,
                f"Letter: {displayed_letter or '-'}",
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
                f"Word: {current_word}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Sentence: {sentence}",
                (30, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 200, 0),
                2,
            )

            cv2.putText(
                frame,
                "SPACE: Add and speak word | ENTER: Speak sentence",
                (30, frame.shape[0] - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                "B: Backspace | R: Reset | Q: Quit",
                (30, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

            cv2.imshow("ASL Real-Time", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord(" "):
                if current_word:
                    word_to_speak = current_word

                    sentence += word_to_speak + " "
                    speech_queue.put(word_to_speak)

                    current_word = ""
                    last_detected_letter = ""
                    stable_count = 0
                    release_count = 0
                    letter_locked = True

            elif key in (10, 13):
                if sentence.strip():
                    speech_queue.put(sentence.strip())

            elif key == ord("b"):
                current_word = current_word[:-1]

            elif key == ord("r"):
                current_word = ""
                sentence = ""

                displayed_letter = ""
                last_detected_letter = ""

                stable_count = 0
                release_count = 0
                letter_locked = False
                confidence = 0.0

    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()

        speech_queue.put(None)
        speech_thread.join(timeout=5)


if __name__ == "__main__":
    main()
    