# 🕵️ Deepfake Detector

A deep learning model to detect AI-generated/fake faces vs real faces.

## 🔗 Live Demo
[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Kaushikigupta/deepfake-detector)

## 🏆 Built for CodeAI Hackathon 2026 — Top 2 Finish

## What I Built
- **Level 1:** Deepfake classifier using EfficientNetB0 + Transfer Learning
  - Trained on 140K real/fake face images
  - 90.8% validation accuracy
- **Level 2:** FGSM adversarial attacks on real images (score: 0.5557, Rank 2)
- **Level 3:** Adversarial training defense (76.7% robust accuracy)

## 🛠️ Tech Stack
Python, TensorFlow, Keras, EfficientNetB0, Gradio, Hugging Face Spaces

## 📁 Files
- `app.py` — Gradio web app
- `requirements.txt` — dependencies
- `deepfake_clean.weights.h5` — trained model weights
