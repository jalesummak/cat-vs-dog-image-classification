import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("date_fruit.h5")

model = load_model()

class_names = ['Ajwa','Galaxy','Medjool','Meneifi','Nabtat Ali','Rutab','Shaishe','Sokari','Sugaey']

st.title("Date Fruit Classification - CNN Model")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((224,224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write("Prediction:", class_names[predicted_class])
    st.write("Confidence: %.2f%%" % (confidence*100))