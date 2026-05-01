import gradio as gr
import numpy as np
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, models
from tf_keras.applications import EfficientNetB0
from tf_keras.optimizers import Adam
import cv2

base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base.trainable = False

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Lambda(lambda x: tf.keras.applications.efficientnet.preprocess_input(x)),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("deepfake_clean.weights.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_deepfake(image):
    image = image.convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_array.shape[1], x + w + pad)
        y2 = min(img_array.shape[0], y + h + pad)
        img_array = img_array[y1:y2, x1:x2]
    img = tf.image.resize(img_array, (224, 224)).numpy().astype(np.float32)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return {
        "🔴 FAKE": float(pred),
        "🟢 REAL": float(1 - pred)
    }

demo = gr.Interface(
    fn=predict_deepfake,
    inputs=gr.Image(type="pil", label="Upload a Face Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="🕵️ Deepfake Detector",
    description="Upload a face image to check if it's **Real** or **AI-generated/Fake**. Built with EfficientNetB0 — 90.8% accuracy. For best results upload PNG/JPG photos directly.",
    theme=gr.themes.Soft(),
    live=False
)

demo.launch()