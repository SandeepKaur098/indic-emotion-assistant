# 🎙️ Indic Emotion Assistant
A speech-based emotion-aware virtual assistant for Indic languages using audio signal processing and machine learning.

## 🔍 Overview
This project aims to build a multilingual, emotion-aware virtual assistant that can detect user emotions from voice inputs in Indic languages. It focuses on low-resource languages like Hindi, Bengali, Marathi, Punjabi, Gujarati, Malayalam, etc., to make emotionally intelligent systems more inclusive and accessible.

## 🎯 Objectives
- Recognize basic human emotions (e.g., happy, sad, angry, neutral) from speech
- Support native languages spoken across India
- Use audio feature extraction techniques and machine learning models
- Build a modular assistant that can respond intelligently based on detected emotion

> 📌 **Note:** This project is currently in development and is being actively coded in **Google Colab** for flexibility and reproducibility.

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Development Environment:** Google Colab, VS Code  
- **Audio Features:** MFCC, Chroma, ZCR, RMS, Mel spectrogram  
- **Libraries & Frameworks:**  
  - `librosa`, `audiomentations`, `soundfile`, `pydub`  
  - `scikit-learn`, `LightGBM`, `pandas`, `numpy`  
  - `matplotlib`, `seaborn` (for visualization)  
- **Tools:** Git, GitHub, Google Drive (for dataset hosting)  
- **(Planned)** TTS Integration: `pyttsx3`, `gTTS`

## 🗂 Project Structure

```bash
indic-emotion-assistant/
├── data/             # Dataset storage (not included - see Drive link)
├── models/           # Trained ML models (.pkl)
├── notebooks/        # Colab notebooks for all stages (augmentation, training)
├── src/              # Python scripts for processing & inference
├── .gitignore        # File ignore rules
├── LICENSE           # MIT license for protection
├── README.md         # You're reading it!
└── requirements.txt  # All Python dependencies
