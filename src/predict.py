import numpy as np
from tensorflow import keras

#Load model
model = keras.models.load_model("../model/digit_model.h5")

#Test with one image from Dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Take one test image
img = x_test[0]
actual = y_test[0]

#Prepare image
img_normalized = img / 255.0
img_flattened = img_normalized.reshape(1, 784)

#Predict
prediction = model.predict(img_flattened)
prediction_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"Actual Digit: {actual}")
print(f"Predicted Digit: {prediction_digit}")
print(f"Confidence: {confidence:.2f}")