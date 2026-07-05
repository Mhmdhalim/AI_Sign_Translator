# ✋ Sign Language Recognition (ASL Project)

This project is a real-time American Sign Language (ASL) recognition system using:
- MediaPipe Hands
- TensorFlow / Keras
- OpenCV
- Scikit-learn

It detects hand landmarks and classifies letters in real-time using a trained neural network.

---

# 🚀 Features

- Real-time hand tracking via webcam
- Landmark extraction (63 features per hand)
- Neural network classification (A-Z signs)
- High accuracy (~99% on test data)
- Works offline after training

---

# 📁 Project Structure

```

ASL_Sign_Translator/
│
├── ASL_dataset/Train/        # Original images dataset
├── landmark_dataset/         # Extracted landmarks (.npy files)
├── models/                   # Trained model + classes
├── utils/                   # Helper scripts
│
├── convert_dataset.py       # Converts images → landmarks
├── train_model.py           # Trains ML model
├── realtime_predict.py      # Live webcam prediction

````

---

# ⚙️ Installation

## 1. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
````

## 2. Install dependencies

⚠️ IMPORTANT: use compatible versions

```bash
pip install tensorflow==2.15.0
pip install mediapipe==0.10.35
pip install protobuf==3.20.3
pip install opencv-python
pip install scikit-learn
pip install numpy
```

---

# 🔄 Workflow

## 1. Convert dataset (images → landmarks)

```bash
python convert_dataset.py
```

Output:

```
landmark_dataset/A/*.npy
landmark_dataset/B/*.npy
```

---

## 2. Train model

```bash
python train_model.py
```

Output:

* Model saved in `models/`
* Accuracy ~95%+

---

## 3. Run real-time prediction

```bash
python realtime_predict.py
```

Press:

* `Q` → quit

---

# 💥 Common Problems & Fixes

---

## ❌ 1. Samples = 0 during training

### Cause:

Training script reading wrong dataset path or missing `.npy` files.

### Fix:

Make sure:

```python
DATASET_PATH = "landmark_dataset"
```

and not `ASL_dataset`.

---

## ❌ 2. Mediapipe error (FieldDescriptor / runtime_version)

### Error:

```
AttributeError: 'FieldDescriptor' has no attribute 'label'
ImportError: runtime_version from protobuf
```

### Cause:

Version mismatch between:

* mediapipe
* tensorflow
* protobuf

### Fix:

```bash
pip uninstall mediapipe tensorflow protobuf -y
pip install tensorflow==2.15.0
pip install mediapipe==0.10.35
pip install protobuf==3.20.3
```

---

## ❌ 3. Mediapipe not detecting hands

### Fix:

* Increase lighting
* Ensure hand is inside camera frame
* Use `static_image_mode=False`

---

## ❌ 4. No `.npy` files generated

### Cause:

MediaPipe not detecting hands in images

### Fix:

Check:

* image clarity
* background contrast
* correct dataset path

---

## ❌ 5. GPU warnings on Windows

```
TensorFlow GPU not available
```

### Fix:

Ignore it (CPU mode works fine)

---

# 🧠 Model Details

* Input: 63 features (x, y, z for 21 landmarks)
* Output: A-Z classes
* Model: Fully Connected Neural Network
* Loss: categorical crossentropy

---

# 📊 Performance

* Training Accuracy: ~99%
* Test Accuracy: ~98–99%

---

# ⚠️ Notes

* Dataset must be converted before training
* Do NOT train directly on images
* Always use landmark_dataset
* Keep dependency versions fixed (very important)

---

# 🚀 Future Improvements

* Real-time word building
* Sentence recognition
* Text-to-speech output
* UI interface (Tkinter / Web App)

---

# 👨‍💻 Author

Built as a full Computer Vision + Deep Learning project for ASL recognition.

