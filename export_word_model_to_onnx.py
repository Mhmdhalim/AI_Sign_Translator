from pathlib import Path

import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "word_classifier.keras"
OUTPUT_PATH = PROJECT_ROOT / "models" / "word_classifier.onnx"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 258


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = load_model(str(MODEL_PATH), compile=False)

    input_signature = (
        tf.TensorSpec(
            shape=(None, SEQUENCE_LENGTH, FEATURE_SIZE),
            dtype=tf.float32,
            name="sequence",
        ),
    )

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=str(OUTPUT_PATH),
    )

    print(f"ONNX model saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()