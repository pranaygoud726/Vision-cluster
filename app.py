import streamlit as st
import base64
import cv2
import numpy as np
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

# Background + UI
st.markdown("""
<style>
body {
    margin: 0;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Card */
.card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 20px;
    width: 350px;
    margin: auto;
    text-align: center;
    box-shadow: 0px 20px 50px rgba(0,0,0,0.3);
    margin-top: 50px;
}

/* Title */
.title {
    text-align:center;
    color:white;
    font-size:40px;
}

/* Button */
button {
    background: #00c6ff;
    color: white;
    border: none;
    padding: 10px 25px;
    border-radius: 10px;
    cursor: pointer;
}

.result {
    margin-top: 20px;
    background: white;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚦 Traffic Sign Clustering</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, width=200)

    with st.spinner("Analyzing..."):
        img_resized = cv2.resize(img, (64,64))
        img_flat = img_resized.flatten().reshape(1,-1)

        kmeans = KMeans(n_clusters=5)
        kmeans.fit(np.random.rand(10, img_flat.shape[1]))

        cluster = kmeans.predict(img_flat)[0]

    st.markdown(f"""
    <div class="result">
    🔍 This looks similar to group <b>{cluster}</b><br>
    📌 Cluster Group: <b>{cluster}</b>
    </div>
    """, unsafe_allow_html=True)
