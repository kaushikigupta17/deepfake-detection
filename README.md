# 🕵️ Deepfake Detector

A deep learning model to detect AI-generated/fake faces vs real faces.

## 🔗 Live Demo
### 👉 [Click here to open the app](https://huggingface.co/spaces/Kaushikigupta/deepfake-detector)
> ⚠️ Open the link directly for best experience — embedded preview may flicker.

## 🏆 Built for CodeAI Hackathon 2026 — Runner up

## What I Built
- **Level 1:** Deepfake classifier using EfficientNetB0 + Transfer Learning — 90.8% accuracy
- **Level 2:** FGSM adversarial attacks on real images — Score: 0.5557
- **Level 3:** Adversarial training defense — 76.7% robust accuracy

## 🧠 Model Architecture
- Transfer Learning from ImageNet weights
- Fine-tuned on 140K real/fake face dataset
- Face auto-detection via OpenCV HaarCascade
- Preprocessing via EfficientNet's built-in normalizer
  
## 🛠️ Tech Stack
Python, TensorFlow, Keras, EfficientNetB0, Gradio, Hugging Face Spaces

## 📁 Files
- `app.py` — Gradio web app
- `requirements.txt` — dependencies
- `deepfake_clean.weights.h5` — trained model weights (on HF Spaces)
