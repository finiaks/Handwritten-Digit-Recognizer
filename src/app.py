import streamlit as st
import numpy as np 
from tensorflow import keras 
from streamlit_drawable_canvas import st_canvas
from PIL import Image

#Load Model
model = keras.models.load_model("model/digit_cnn_model.keras")

#Title
st.title("Handwritten Digit Recognizer")
st.write("Draw a Digit (0 - 9) and the AI will predict it!")

#Drawing Canvas
canvas = st_canvas(
    fill_color = "black",
    stroke_width = 15,
    stroke_color = "white",
    background_color = "black",
    height = 280,
    width = 280,
    drawing_mode = "freedraw",
    key = "canvas"
)

#Predict Button
if st.button("Predict Digit!"):
    if canvas.image_data is not None:
        #Process image
        img = canvas.image_data
        img = Image.fromarray(img.astype('uint8'))
        img = img.convert('L')
        img = img.resize((28,28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        #Predict 
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Prediction Digit: {digit}")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please draw a digit first!")
