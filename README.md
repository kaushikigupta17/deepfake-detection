# Deepfake Detection - AI vs AI Hackathon

## Team Kasukabe
- Kaushiki Gupta (@kaushikigupta17)
- Bhumi Mittal (@Bhumimittal15)

## Overview
A complete AI pipeline for deepfake detection, adversarial attacks, and robust defense.
Built for CodeAI Hackathon 2026 — JIIT Noida.

## Round 1 — Detection
- Model: EfficientNetB0 with Transfer Learning
- Dataset: 140,000 face images (real vs fake)
- Validation Accuracy: **89.5%**
- Test Predictions: 20,000 images classified

## Round 2 — Attack
- Method: FGSM (Fast Gradient Sign Method)
- Attack Score: **0.5557**
- Ranking: **2nd Place — Qualified for Finals**

## Round 3 — Defense
- Method: Adversarial Training (50/50 clean + attacked images)
- Robust Accuracy: **76.7%**

## Files
- `Untitled0.ipynb` — Full training notebook
- `predictions.csv` — Level 1 test predictions
- `attacked_images.zip` — Level 2 adversarial images
- `predictions_robust.csv` — Level 3 robust predictions
- `deepfake_model_robust.h5` — Final trained model

## Tech Stack
Python, TensorFlow, Keras, EfficientNetB0, Google Colab, Kaggle
