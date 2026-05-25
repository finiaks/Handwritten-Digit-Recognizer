import numpy as np
import tensorflow as tf
from tensorflow import keras

#Load Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

#Flatten 
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

#Improved Model
model = keras.Sequential([
    #Layer 1
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    #Layer 2
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    #Layer 3
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dropout(0.2),

    #Output
    keras.layers.Dense(10, activation = 'softmax')
])

#Compile
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#Train
history = model.fit(
    x_train, y_train,
    epochs = 15,
    batch_size = 32,
    validation_split = 0.2
)

#Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("\nPrevious Accuracy: 97.5%")
print(f"New Accuracy: {test_accuracy * 100: .2f}%")

#Save
model.save("../model/digit_model_improved.keras")
