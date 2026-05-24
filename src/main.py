import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Load MNIST Dataset
(x_train, y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

print("Training samples:", x_train.shape)
print("Testing samples:", x_test.shape)
print("Sample label:", y_train[0])

#Show one image
plt.imshow(x_train[0], cmap = 'gray')
plt.title(f"label: {y_train[0]}")
plt.show()

#Normalize pixels (0 - 255 -> 0 - 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

#Flatten images (28 x 28 -> 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print("After reshape:", x_train.shape)

#Build Neural Network
model = keras.Sequential([
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

#Compile
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'] 
)

#Train
print("\nTraining started....")
history = model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)

#Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", test_accuracy)

#Save Model
model.save("../model/digit_model.h5")