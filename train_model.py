import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

DATASET_PATH = "landmark_dataset"

X = []
y = []

for label in os.listdir(DATASET_PATH):

    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):

        if not file.endswith(".npy"):
            continue

        data = np.load(os.path.join(label_path, file))

        X.append(data)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Samples:", len(X))
print("Shape:", X.shape)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

num_classes = len(np.unique(y))

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(63,)))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))

model.add(Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy:.4f}")

os.makedirs("models", exist_ok=True)

model.save("models/sign_classifier.keras")

np.save("models/classes.npy", encoder.classes_)

print("Model saved successfully.")