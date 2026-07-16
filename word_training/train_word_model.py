from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GRU,
    Input,
)
from tensorflow.keras.models import Sequential


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LANDMARKS_DIR = PROJECT_ROOT / "word_training" / "landmarks"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "word_classifier.keras"
BEST_MODEL_PATH = MODELS_DIR / "word_classifier_best.keras"
CLASSES_PATH = MODELS_DIR / "word_classes.npy"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 258
BATCH_SIZE = 32
EPOCHS = 60
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def get_class_names():
    class_names = set()

    for split_name in ("train", "val", "test"):
        split_path = LANDMARKS_DIR / split_name

        if not split_path.exists():
            continue

        for class_path in split_path.iterdir():
            if class_path.is_dir():
                class_names.add(class_path.name)

    if not class_names:
        raise RuntimeError(
            f"No class folders were found inside: {LANDMARKS_DIR}"
        )

    return sorted(class_names)


def load_split(split_name, class_to_index):
    split_path = LANDMARKS_DIR / split_name

    if not split_path.exists():
        print(f"Split not found: {split_path}")
        return (
            np.empty(
                (0, SEQUENCE_LENGTH, FEATURE_SIZE),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int32),
        )

    samples = []
    labels = []
    invalid_files = 0

    for class_name, class_index in class_to_index.items():
        class_path = split_path / class_name

        if not class_path.exists():
            continue

        for file_path in class_path.glob("*.npy"):
            try:
                sequence = np.load(file_path).astype(np.float32)

                if sequence.shape != (
                    SEQUENCE_LENGTH,
                    FEATURE_SIZE,
                ):
                    invalid_files += 1
                    continue

                if not np.isfinite(sequence).all():
                    invalid_files += 1
                    continue

                samples.append(sequence)
                labels.append(class_index)

            except Exception as error:
                invalid_files += 1
                print(f"Skipped {file_path}: {error}")

    if not samples:
        return (
            np.empty(
                (0, SEQUENCE_LENGTH, FEATURE_SIZE),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int32),
        )

    x = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)

    print(
        f"{split_name}: {len(x)} samples, "
        f"{invalid_files} invalid files"
    )

    return x, y


def build_model(number_of_classes):
    model = Sequential(
        [
            Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
            GRU(
                128,
                return_sequences=True,
                dropout=0.2,
            ),
            BatchNormalization(),
            GRU(
                64,
                return_sequences=False,
                dropout=0.2,
            ),
            BatchNormalization(),
            Dense(128, activation="relu"),
            Dropout(0.4),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(number_of_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    if not LANDMARKS_DIR.exists():
        raise FileNotFoundError(
            f"Landmarks directory was not found: {LANDMARKS_DIR}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names()
    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    print(f"Classes: {len(class_names)}")

    x_train, y_train = load_split(
        "train",
        class_to_index,
    )

    x_val, y_val = load_split(
        "val",
        class_to_index,
    )

    x_test, y_test = load_split(
        "test",
        class_to_index,
    )

    if len(x_train) == 0:
        raise RuntimeError(
            "The training split contains no valid samples."
        )

    if len(x_val) == 0:
        print(
            "Validation split is empty. "
            "Using 20% of the training samples."
        )

        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        validation_size = max(
            1,
            int(len(indices) * 0.2),
        )

        validation_indices = indices[:validation_size]
        training_indices = indices[validation_size:]

        x_val = x_train[validation_indices]
        y_val = y_train[validation_indices]

        x_train = x_train[training_indices]
        y_train = y_train[training_indices]

    print(f"Training shape: {x_train.shape}")
    print(f"Validation shape: {x_val.shape}")
    print(f"Test shape: {x_test.shape}")

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=0.00001,
        ),
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
    )

    if len(x_test) > 0:
        test_loss, test_accuracy = model.evaluate(
            x_test,
            y_test,
            verbose=1,
        )

        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        probabilities = model.predict(
            x_test,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        used_labels = sorted(
            set(y_test.tolist())
            | set(predictions.tolist())
        )

        used_names = [
            class_names[index]
            for index in used_labels
        ]

        print(
            classification_report(
                y_test,
                predictions,
                labels=used_labels,
                target_names=used_names,
                zero_division=0,
            )
        )
    else:
        print("The test split is empty. Evaluation was skipped.")

    model.save(MODEL_PATH)
    np.save(CLASSES_PATH, np.asarray(class_names))

    print(f"Model saved: {MODEL_PATH}")
    print(f"Classes saved: {CLASSES_PATH}")


if __name__ == "__main__":
    main()