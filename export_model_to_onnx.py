from pathlib import Path

import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import load_model


MODEL_PATH = Path("models/sign_classifier.keras")
OUTPUT_PATH = Path("models/sign_classifier.onnx")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = load_model(MODEL_PATH, compile=False)

    input_signature = (
        tf.TensorSpec(
            shape=(None, 63),
            dtype=tf.float32,
            name="landmarks",
        ),
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=str(OUTPUT_PATH),
    )

    print(f"ONNX model saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()