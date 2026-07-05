import cv2
import time
import numpy as np

class Recorder:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.recording = False
        self.frames = []

    def countdown(self, frame):
        for i in range(3, 0, -1):
            start_time = time.time()
            while time.time() - start_time < 1:
                cv2.putText(
                    frame,
                    f"Starting in {i}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )
                cv2.imshow("Recorder", frame)
                cv2.waitKey(1)

    def start(self):
        self.recording = True
        self.frames = []

    def stop(self):
        self.recording = False

    def add_frame(self, data):
        if self.recording:
            self.frames.append(data)

    def is_full(self):
        return len(self.frames) >= self.sequence_length
        def get_data(self):
            return np.array(self.frames)

    def reset(self):
        self.frames = []
        self.recording = False

    def play_back(self):
        for frame in self.frames:
            cv2.imshow("Preview Recording", frame)

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord("y"):
                    return True

                if key == ord("n"):
                    return False

        cv2.destroyWindow("Preview Recording")

    def save(self, path, video_path=None):
        np.save(path, np.array(self.frames))
        print(f"[INFO] Saved NPY: {path}")

        if video_path is not None:
            import shutil
            video_save_path = path.replace(".npy", ".mp4")
            shutil.move(video_path, video_save_path)
            print(f"[INFO] Saved MP4: {video_save_path}")

    def record_sequence(self, cap, hands, mp_draw, mp_hands):
        """
        Full pipeline:
        countdown → record → preview → save or discard
        """

        # Countdown (3..2..1)
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        self.countdown(frame)

        print("[INFO] Recording started...")

        self.start()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        video_path = None
        
        while len(self.frames) < self.sequence_length:

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            if out is None:
                h, w, _ = frame.shape
                video_path = "temp_recording.mp4"
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

            # if only one hand → fill zeros
            while len(landmarks) < 126:
                landmarks.append(0)

            self.add_frame(landmarks)

            cv2.putText(
                frame,
                f"Recording... {len(self.frames)}/{self.sequence_length}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            cv2.imshow("Recorder", frame)
            out.write(frame)
            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.reset()
                return False

        print("[INFO] Recording finished")
        out.release()

        # Preview
        print("[INFO] Sequence recorded successfully (no preview for landmark data)")

        # Ask user
        print("Save this sequence? (y/n): ")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("y"):
                return True, video_path

            elif key == ord("n"):
                self.reset()
                return False