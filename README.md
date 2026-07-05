# 🤖 AI Sign Language Translator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0-red.svg)](https://mediapipe.dev)

> Real-time sign language translation to text and speech using AI

---

## ⚠️ IMPORTANT NOTES (Read First!)

Before you start, make sure you understand these **critical requirements**:

### 🔴 Mandatory Requirements

| Requirement | Why? | How? |
|-------------|------|------|
| **Python 3.8+** | MediaPipe and OpenCV require Python 3.8 or higher | Check: `python --version` |
| **Dataset** | The app won't work without data | Download from [link] and place in `dataset/` |
| **Virtual Environment** | Prevents library conflicts | Create with `python -m venv env` |
| **MediaPipe Version** | Must use MediaPipe 0.10.0+ | `pip install mediapipe==0.10.0` |
| **Standard Webcam** | Any built-in or USB webcam works | Built-in laptop camera or USB webcam |
| **Git LFS** | Only if you plan to upload large files | `git lfs install` |

---

## 📥 Mandatory Dataset Download

> **⚠️ CRITICAL:** The app will **NOT** work without the dataset!

You **MUST** download the dataset before running the application:

1. **Download from:**  
   👉 [https://data.mendeley.com/public-api/zip/jdyksv2jhh/download/1](https://data.mendeley.com/public-api/zip/jdyksv2jhh/download/1)

2. **Extract** the ZIP file

3. **Rename the extracted folder** to `dataset` and place it in the project root: 
> **❌ If the dataset is missing, you will get an error like:**  
> `FileNotFoundError: dataset/A/sample1.jpg not found`

---

## 📸 Camera Setup

> **✅ Any standard webcam works!**  
> - Built-in laptop camera ✅  

**No special camera required.** Just make sure:
1. Your camera is connected (if external)
2. Camera permissions are enabled
3. No other app is using the camera

**To test your camera:**
```python
# Run this in Python to test
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
 print("✅ Camera working!")
else:
 print("❌ Camera not found")
cap.release()