import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN (4D input)
# (samples, height , width , channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Shape for CNN: ", x_train.shape)

#Build CNN Model
model = keras.Sequential([
    # First Convolution Block
    keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),

    # Second Convolution Block
    keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),

    # Third Convolution Block
    keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    keras.layers.BatchNormalization(),

    # Flatten and Dense
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation = 'softmax')
])

# Model Summary
model.summary()

# Compile
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# Train 
history = model.fit(
    x_train, y_train,
    epochs = 15,
    batch_size = 32,
    validation_split = 0.2
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\n Basic NN Accuracy: 97.5%")
print("\n Imporved NN Accuracy: 98.03%")
print(f"CNN Accuracy: {test_accuracy * 100:.2f}%")

# Save Model
model.save("../model/digit_cnn_model.keras")