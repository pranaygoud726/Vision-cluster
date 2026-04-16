import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans

st.set_page_config(page_title="Vision Cluster", layout="centered")

st.title("🚦 Traffic Sign Clustering")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = cv2.resize(img, (64, 64))
    img_flat = img_resized.flatten().reshape(1, -1)

    # Dummy clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(np.random.rand(10, img_flat.shape[1]))

    cluster = kmeans.predict(img_flat)[0]

    st.success(f"🔍 This looks similar to group {cluster}")
    st.info(f"📌 Cluster Group: {cluster}")
