# Handwritten Digit Recognizer

A Deep Learning project that recognizes handwritten
digits (0-9) using Neural Networks trained on the
MNIST dataset.

## Project Overview

Draw any digit on the canvas and the AI predicts
it instantly with confidence score!

## How It Works

1. User draws a digit on canvas
2. Image resized to 28x28 pixels
3. Neural Network analyzes 784 pixels
4. Predicts digit with confidence score

## Tech Stack

- Python
- TensorFlow & Keras (Deep Learning)
- MNIST Dataset (60,000 images)
- Streamlit (Web UI)
- Streamlit Drawable Canvas

## Model Performance

| Model                   | Accuracy |
| ----------------------- | -------- |
| Basic Neural Network    | 97.5%    |
| Improved Neural Network | 98.03%   |

## 🔧 Model Improvements

- Added Dropout(0.3) → prevents overfitting
- Added BatchNormalization → faster stable training
- Increased neurons 128 → 256
- Increased epochs 10 → 15

## 🏗️ Architecture

### Basic Model

Input(784) → Dense(128,ReLU) → Dense(64,ReLU) → Output(10,Softmax)

### Improved Model

Input(784) → Dense(256,ReLU) → BatchNorm → Dropout(0.3)
→ Dense(128,ReLU) → BatchNorm → Dropout(0.3)
→ Dense(64,ReLU) → Dropout(0.2)
→ Output(10,Softmax)

## Project Structure

digit_recognition_ml/
├── model/
│ └── digit_model.h5
│ └── digit_model_improved.keras
├── src/
│ ├── main.py
│ └── predict.py
│ └── improved_model.py
│ └── app.py
└── requirements.txt

## Live Demo

This project runs locally due to TensorFlow
compatibility with Streamlit Cloud.

## How to Run Locally

1. Clone the repo
2. Install requirements
   pip install -r requirements.txt
3. Run the app
   streamlit run src/app.py
