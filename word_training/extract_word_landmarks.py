import json
import re
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "word_training"
    / "manifests"
    / "wlasl100_manifest.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "word_training"
    / "landmarks"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "word_training"
    / "landmark_extraction_report.json"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 258

MIN_VIDEO_SIZE = 1024

SKIP_EXISTING = True

# Set this to 20 for a quick test.
# Set it to None to process all videos.
MAX_VIDEOS = None


mp_holistic = mp.solutions.holistic


def sanitize_name(value):
    cleaned_value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return cleaned_value.strip("_").lower()


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest file was not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, list):
        raise ValueError("Manifest must contain a list of video samples.")

    return manifest


def get_frame_indices(frame_count):
    if frame_count <= 0:
        return []

    if frame_count == 1:
        return [0] * SEQUENCE_LENGTH

    indices = np.linspace(
        0,
        frame_count - 1,
        SEQUENCE_LENGTH,
    )

    return indices.astype(int).tolist()


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
    pose_features = extract_pose_features(
        results.pose_landmarks
    )

    left_hand_features = extract_hand_features(
        results.left_hand_landmarks
    )

    right_hand_features = extract_hand_features(
        results.right_hand_landmarks
    )

    frame_features = np.concatenate(
        [
            pose_features,
            left_hand_features,
            right_hand_features,
        ]
    )

    if frame_features.shape != (FEATURE_SIZE,):
        raise ValueError(
            f"Unexpected feature shape: {frame_features.shape}"
        )

    return frame_features


def read_selected_frame(capture, frame_index):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = capture.read()

    if not success or frame is None:
        return None

    return frame


def extract_video_sequence(video_path, holistic):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return None, "could_not_open_video"

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_indices = get_frame_indices(frame_count)

    if not frame_indices:
        capture.release()
        return None, "video_has_no_frames"

    sequence = []

    try:
        for frame_index in frame_indices:
            frame = read_selected_frame(
                capture,
                frame_index,
            )

            if frame is None:
                sequence.append(
                    np.zeros(
                        FEATURE_SIZE,
                        dtype=np.float32,
                    )
                )
                continue

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)

            frame_features = extract_frame_features(
                results
            )

            sequence.append(frame_features)

    finally:
        capture.release()

    sequence_array = np.asarray(
        sequence,
        dtype=np.float32,
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURE_SIZE,
    )

    if sequence_array.shape != expected_shape:
        return (
            None,
            f"invalid_sequence_shape_{sequence_array.shape}",
        )

    if not np.isfinite(sequence_array).all():
        return None, "sequence_contains_invalid_values"

    return sequence_array, None


def build_output_path(item):
    split = sanitize_name(
        str(item.get("split", "unknown"))
    )

    gloss = sanitize_name(
        str(item.get("gloss", "unknown"))
    )

    video_id = sanitize_name(
        str(item.get("video_id", "unknown"))
    )

    return (
        OUTPUT_DIR
        / split
        / gloss
        / f"{video_id}.npy"
    )


def resolve_video_path(item):
    relative_path = item.get("video_path")

    if not relative_path:
        return None

    return PROJECT_ROOT / relative_path


def save_report(report):
    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Report saved: {REPORT_PATH}")


def main():
    manifest = load_manifest()

    if MAX_VIDEOS is not None:
        manifest = manifest[:MAX_VIDEOS]

    print(f"Manifest samples: {len(manifest)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Feature size: {FEATURE_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    counters = Counter()
    failed_samples = []

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
    ) as holistic:

        for index, item in enumerate(manifest, start=1):
            video_id = str(
                item.get("video_id", "unknown")
            )

            gloss = str(
                item.get("gloss", "unknown")
            )

            video_path = resolve_video_path(item)
            output_path = build_output_path(item)

            if (
                SKIP_EXISTING
                and output_path.exists()
            ):
                counters["skipped_existing"] += 1

                print(
                    f"[{index}/{len(manifest)}] "
                    f"Skipped existing: {gloss} / {video_id}"
                )
                continue

            if video_path is None:
                counters["failed"] += 1

                failed_samples.append(
                    {
                        "video_id": video_id,
                        "gloss": gloss,
                        "reason": "missing_video_path",
                    }
                )
                continue

            if not video_path.exists():
                counters["failed"] += 1

                failed_samples.append(
                    {
                        "video_id": video_id,
                        "gloss": gloss,
                        "reason": "video_file_not_found",
                        "video_path": str(video_path),
                    }
                )

                print(
                    f"[{index}/{len(manifest)}] "
                    f"Missing video: {video_path}"
                )
                continue

            if video_path.stat().st_size < MIN_VIDEO_SIZE:
                counters["failed"] += 1

                failed_samples.append(
                    {
                        "video_id": video_id,
                        "gloss": gloss,
                        "reason": "video_file_too_small",
                        "video_path": str(video_path),
                    }
                )

                print(
                    f"[{index}/{len(manifest)}] "
                    f"Invalid video: {video_path}"
                )
                continue

            sequence, error = extract_video_sequence(
                video_path,
                holistic,
            )

            if error is not None:
                counters["failed"] += 1

                failed_samples.append(
                    {
                        "video_id": video_id,
                        "gloss": gloss,
                        "reason": error,
                        "video_path": str(video_path),
                    }
                )

                print(
                    f"[{index}/{len(manifest)}] "
                    f"Failed: {gloss} / {video_id} / {error}"
                )
                continue

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.save(
                output_path,
                sequence,
            )

            counters["saved"] += 1

            print(
                f"[{index}/{len(manifest)}] "
                f"Saved: {gloss} / {video_id}"
            )

    report = {
        "manifest_samples": len(manifest),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_size": FEATURE_SIZE,
        "saved": counters["saved"],
        "skipped_existing": counters["skipped_existing"],
        "failed": counters["failed"],
        "failed_samples": failed_samples,
    }

    save_report(report)

    print()
    print("Landmark extraction completed.")
    print(f"Saved: {counters['saved']}")
    print(
        f"Skipped existing: "
        f"{counters['skipped_existing']}"
    )
    print(f"Failed: {counters['failed']}")


if __name__ == "__main__":
    main()