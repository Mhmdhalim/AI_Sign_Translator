import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for index, landmark in enumerate(hand_landmarks.landmark):
                height, width, _ = frame.shape

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.putText(
                    frame,
                    str(index),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1
                )

                print(f"Landmark {index}: ({landmark.x:.3f}, {landmark.y:.3f}, {landmark.z:.3f})")

    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()