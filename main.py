import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.header("Welcome to Deep Neural Network for MNIST Classification")
st.text("")
st.text("""We'll apply all the knowledge from the lectures in this section to write a deep neural network. The problem we've chosen is referred to as the "Hello World" of deep learning because for most students it is the first deep learning algorithm they see.

The dataset is called MNIST and refers to handwritten digit recognition. You can find more about it on Yann LeCun's website (Director of AI Research, Facebook). He is one of the pioneers of what we've been talking about and of more complex approaches that are widely used today, such as covolutional neural networks (CNNs).

The dataset provides 70,000 images (28x28 pixels) of handwritten digits (1 digit per image).

The goal is to write an algorithm that detects which digit is written. Since there are only 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), this is a classification problem with 10 classes.

Our goal would be to build a neural network with 2 hidden layers.""")
st.text("")

# Load model
new_model = tf.keras.models.load_model('MNIST_model.h5')

# Load anh len
ori_img = st.file_uploader("Choose MNIST image (Pleaseee only choose the MNIST digit image")
st.image(ori_img, caption="Input image")
if not (ori_img is None):
    # Xu ly anh
    img = Image.open(ori_img).convert("L").resize((28, 28))
    x = tf.keras.preprocessing.image.img_to_array(img)[np.newaxis, ...]
    print(x.shape)
    preds = new_model.predict(x)
    for i in range(len(preds[0])):
        if (preds[0][i] == 1):
            st.write("Predict number is: ", i)
