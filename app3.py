import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
# MobileNetV2 için gerekli ön işleme fonksiyonu
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Hurma Transfer Learning", page_icon="🌴")

@st.cache_resource
def load_assets():
    model_path = 'date_fruit_transferlearning.h5'
    # compile=False hatayı önlemek için çok kritiktir
    model = tf.keras.models.load_model(model_path, compile=False)
    
    labels = [
        'Galaxy', 'Rutab', 'Sugaey', 'Medjool', 'Nabtat Ali', 
        'Ajwa', 'Sokari', 'Shaishe', 'Meneifi'
    ]
    return model, labels

model, labels = load_assets()

st.title("🌴 Transfer Learning - Hurma Tanımlayıcı")

img_file = st.camera_input("Hurmanın fotoğrafını çekin")
uploaded_file = st.file_uploader("Veya yükleyin", type=["jpg", "png", "jpeg"])

target_file = img_file if img_file is not None else uploaded_file

if target_file:
    image = Image.open(target_file).convert('RGB')
    st.image(image, caption='İşlenen Resim', use_container_width=True)
    
    # EĞİTİM KODUNA GÖRE GÜNCELLEME:
    # Görseldeki eğitim kodunda img_size = 224 olarak belirlenmiş.
    img = image.resize((224, 224)) 
    img_array = np.array(img)
    
    # Görseldeki eğitim kodunda preprocess_input kullanılmış. 
    # Sadece 255'e bölmek yerine bu fonksiyonu kullanmak doğruluğu artırır.
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    with st.spinner('Model analiz ediyor...'):
        # predict_on_batch bazen çoklu tensor hatalarını çözebilir
        prediction = model.predict(img_array)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    st.success(f"### Tahmin: {labels[class_id]}")
    st.info(f"**Doğruluk Payı:** %{confidence:.2f}")
    st.progress(min(int(confidence), 100))