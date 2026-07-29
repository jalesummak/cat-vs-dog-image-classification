import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Sayfa yapılandırması
st.set_page_config(page_title="Hurma Çeşidi Sınıflandırma", page_icon="🌴")

@st.cache_resource
def load_assets():
    # Modelini yüklüyoruz
    model = tf.keras.models.load_model('date_fruit.h5')
    
    # Pickle dosyan olmadığı için sözlüğü manuel tanımlıyoruz
    # Modelin çıktı indekslerine göre sıralı liste:
    labels = [
        'Galaxy', 'Rutab', 'Sugaey', 'Medjool', 'Nabtat Ali', 
        'Ajwa', 'Sokari', 'Shaishe', 'Meneifi'
    ]
    return model, labels

model, labels = load_assets()

# Arayüz başlıkları
st.title("🌴 Hurma Çeşidi Tanımlayıcı")
st.write("Bir hurma fotoğrafı çekin veya yükleyin, hangi çeşit olduğunu tahmin edelim.")

# Giriş seçenekleri
img_file = st.camera_input("Hurmanın fotoğrafını çekin")
uploaded_file = st.file_uploader("Veya bir fotoğraf yükleyin", type=["jpg", "png", "jpeg"])

target_file = img_file if img_file is not None else uploaded_file

if target_file:
    # Görüntüyü açma ve gösterme
    image = Image.open(target_file)
    st.image(image, caption='İşlenen Fotoğraf', use_container_width=True)
    
    # ÖNEMLİ: Ekran görüntündeki model (170, 170) giriş boyutuna sahip
    img = image.resize((170, 170)) 
    img_array = np.array(img) / 255.0  # Normalize etme
    
    # Modelin beklediği 4 boyutlu yapıya getirme (Batch size ekleme)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Analiz ediliyor...'):
        prediction = model.predict(img_array)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction) * 100

    # Sonuçları ekrana yazdırma
    st.success(f"### Tahmin: {labels[class_id]}")
    st.write(f"**Doğruluk Payı:** %{confidence:.2f}")
    st.progress(int(confidence))