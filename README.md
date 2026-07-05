# 🤖 AI Sign Language Translator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0-red.svg)](https://mediapipe.dev)

> Real-time sign language translation to text and speech using AI

---

## 📖 About

This project uses **MediaPipe** and **OpenCV** to translate hand gestures (sign language) into written text and spoken audio in real-time.

### Features
- 🖐️ High-accuracy hand gesture detection
- 📝 Real-time sign language to text translation
- 🔊 Text-to-speech conversion
- 📸 Live webcam support
- 💾 Custom sign data collection

---

## 🚀 Requirements

### System Requirements
- Windows 10/11, Linux, or macOS
- Python 3.8 or higher
- Webcam
- Internet connection (for downloading libraries)

### Required Libraries
```txt
opencv-python
mediapipe
numpy
pyttsx3
```

---

## 🔧 Installation & Setup (Step-by-Step)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Mhmdhalim/AI_Sign_Translator.git
cd AI_Sign_Translator
```

### Step 2: Create a Virtual Environment
```bash
# Windows
python -m venv env
.\env\Scripts\activate

# Linux/macOS
python3 -m venv env
source env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset
> **⚠️ Important:** The dataset is not included in this repository due to size limits.

You have two options:

#### Option A: Collect Your Own Data
Run the data collection script to capture your own sign language gestures:
```bash
python collect_data.py
```

#### Option B: Download the Dataset
Download the pre-collected dataset from [Google Drive Link] and extract it into the `dataset/` folder.

### Step 5: Run the Application
```bash
python main.py
```

---

## 📁 Project Structure

```
AI_Sign_Translator/
├── dataset/                 # Sign language data (not included in repo)
│   └── ...
├── animations/              # UI animations
├── assets/                  # Images and assets
├── sounds/                  # Sound effects
├── utils/                   # Utility functions
├── camera.py                # Camera handling
├── collect_data.py          # Data collection script
├── collect_sequences.py     # Sequence collection
├── hand_detector.py         # Hand detection logic
├── hand_landmarks.py        # Hand landmarks extraction
├── main.py                  # Main application entry point
├── speech.py                # Text-to-speech functionality
├── translator.py            # Sign translation logic
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files/folders
└── README.md                # This file
```

---

## 🎮 How to Use

1. **Collect Data** (first time only):
   ```bash
   python collect_data.py
   ```
   Follow the on-screen instructions to capture hand gestures for each sign.

2. **Run the Translator**:
   ```bash
   python main.py
   ```
   - The webcam will open
   - Show hand signs to the camera
   - The system will detect and translate them in real-time
   - Text and audio output will be displayed

3. **Customize**:
   - Add new signs by running `collect_data.py` again
   - Adjust detection settings in `hand_detector.py`

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `Module not found` | Run `pip install -r requirements.txt` |
| No dataset folder | Create `dataset/` folder and add data |
| Camera not working | Check webcam permissions |
| Git LFS issues | Install Git LFS: `git lfs install` |

---

## 📊 Dataset Information

- The dataset consists of hand landmark coordinates for various sign language gestures
- Each sign has multiple samples for better accuracy
- Format: `.npy` or `.json` files containing hand keypoints

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Commit: `git commit -m "Add your feature"`
5. Push: `git push origin feature/your-feature`
6. Open a Pull Request

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Mhmdhalim**  
GitHub: [@Mhmdhalim](https://github.com/Mhmdhalim)

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev) for hand tracking
- [OpenCV](https://opencv.org) for computer vision
- [pyttsx3](https://pyttsx3.readthedocs.io) for text-to-speech
```

---

## 📌 Notes for You:

| What to Change | Where |
|----------------|-------|
| Add your Google Drive link (if you upload the dataset) | Under "Option B: Download the Dataset" |
| Add your license file | Create `LICENSE` file if you want |
| Add screenshots/videos | Create `screenshots/` folder and add images |

---

**Want me to add anything specific?** Like:
- Screenshots section?
- Demo video link?
- Troubleshooting for common issues?
- More detailed dataset format explanation?

Just tell me what you need! 🚀