import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WLASL_ROOT = (
    PROJECT_ROOT
    / "external_datasets"
    / "WLASL"
    / "start_kit"
)

JSON_PATH = WLASL_ROOT / "WLASL_v0.3.json"
VIDEOS_DIR = WLASL_ROOT / "videos"

OUTPUT_DIR = PROJECT_ROOT / "word_training" / "manifests"

TOP_K = 20
MIN_SAMPLES_PER_CLASS = 5


def validate_paths() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"WLASL JSON file was not found: {JSON_PATH}"
        )

    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(
            f"Processed videos directory was not found: {VIDEOS_DIR}"
        )


def load_wlasl_data() -> list[dict]:
    with JSON_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("WLASL JSON must contain a list of gloss entries.")

    return data


def get_existing_video_ids() -> set[str]:
    return {
        video_path.stem
        for video_path in VIDEOS_DIR.glob("*.mp4")
        if video_path.stat().st_size >= 1024
    }


def build_manifest(
    dataset: list[dict],
    existing_video_ids: set[str],
) -> tuple[list[dict], list[str], Counter]:
    candidates = []

    for gloss_entry in dataset:
        gloss = str(gloss_entry.get("gloss", "")).strip()

        if not gloss:
            continue

        valid_instances = []

        for instance in gloss_entry.get("instances", []):
            video_id = str(instance.get("video_id", "")).strip()

            if not video_id or video_id not in existing_video_ids:
                continue

            video_path = VIDEOS_DIR / f"{video_id}.mp4"

            valid_instances.append(
                {
                    "video_id": video_id,
                    "gloss": gloss,
                    "split": str(instance.get("split", "unknown")),
                    "signer_id": instance.get("signer_id"),
                    "instance_id": instance.get("instance_id"),
                    "variation_id": instance.get("variation_id"),
                    "fps": instance.get("fps", 25),
                    "video_path": str(
                        video_path.relative_to(PROJECT_ROOT)
                    ),
                }
            )

        if len(valid_instances) >= MIN_SAMPLES_PER_CLASS:
            candidates.append((gloss, valid_instances))

    candidates.sort(
        key=lambda item: len(item[1]),
        reverse=True,
    )

    selected_candidates = candidates[:TOP_K]

    manifest = []
    selected_classes = []
    class_counts = Counter()

    for gloss, instances in selected_candidates:
        selected_classes.append(gloss)
        manifest.extend(instances)
        class_counts[gloss] = len(instances)

    return manifest, selected_classes, class_counts

def save_manifest(
    manifest: list[dict],
    selected_classes: list[str],
    class_counts: Counter,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "wlasl20_manifest.json"
    classes_path = OUTPUT_DIR / "wlasl20_classes.txt"
    summary_path = OUTPUT_DIR / "wlasl20_summary.csv"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    with classes_path.open("w", encoding="utf-8") as file:
        for class_name in selected_classes:
            file.write(f"{class_name}\n")

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["class", "sample_count"])

        for class_name in selected_classes:
            writer.writerow(
                [
                    class_name,
                    class_counts[class_name],
                ]
            )

    print(f"Manifest saved: {manifest_path}")
    print(f"Classes saved: {classes_path}")
    print(f"Summary saved: {summary_path}")


def print_summary(
    manifest: list[dict],
    selected_classes: list[str],
    class_counts: Counter,
) -> None:
    split_counts = Counter(
        item["split"]
        for item in manifest
    )

    print()
    print("WLASL preparation completed.")
    print(f"Selected classes: {len(selected_classes)}")
    print(f"Selected videos: {len(manifest)}")
    print(f"Train videos: {split_counts.get('train', 0)}")
    print(f"Validation videos: {split_counts.get('val', 0)}")
    print(f"Test videos: {split_counts.get('test', 0)}")

    if selected_classes:
        print()
        print("First selected classes:")

        for class_name in selected_classes[:10]:
            print(
                f"- {class_name}: "
                f"{class_counts[class_name]} videos"
            )


def main() -> None:
    validate_paths()

    print(f"Reading dataset metadata from: {JSON_PATH}")
    dataset = load_wlasl_data()

    print(f"Checking processed videos in: {VIDEOS_DIR}")
    existing_video_ids = get_existing_video_ids()

    print(f"Valid processed videos found: {len(existing_video_ids)}")

    manifest, selected_classes, class_counts = build_manifest(
        dataset,
        existing_video_ids,
    )

    if not manifest:
        raise RuntimeError(
            "No usable WLASL samples were found. "
            "Check the videos directory and JSON file."
        )

    save_manifest(
        manifest,
        selected_classes,
        class_counts,
    )

    print_summary(
        manifest,
        selected_classes,
        class_counts,
    )


if __name__ == "__main__":
    main()