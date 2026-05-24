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

- Training Accuracy: 99.4%
- Test Accuracy: 97.5%
- Architecture: 784 → 128 → 64 → 10

## Project Structure

digit_recognition_ml/
├── model/
│ └── digit_model.h5
├── src/
│ ├── main.py
│ └── predict.py
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
