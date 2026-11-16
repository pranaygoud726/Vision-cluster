import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from sklearn.cluster import KMeans

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Vision-Cluster Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F8FAFC;
}
[data-testid="stSidebar"] {
    background-color: #172235;
}
.metric-card {
    background: #1E2B40;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #34445D;
}
.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #60A5FA;
}
.sub-title {
    color: #AAB8CC;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- CONSTANTS ----------------

DATASET_DIR = Path("dataset")
IMAGE_SIZE = (32, 32)
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------- DATASET LOADING ----------------

@st.cache_data
def load_dataset():
    images = []
    names = []

    if not DATASET_DIR.exists():
        return np.empty((0, 32, 32, 3)), []

    for file_path in DATASET_DIR.rglob("*"):
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            continue

        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)

            images.append(np.asarray(img, dtype=np.float32) / 255.0)
            names.append(str(file_path.relative_to(DATASET_DIR)))

        except Exception:
            continue

    if not images:
        return np.empty((0, 32, 32, 3)), []

    return np.stack(images), names


# ---------------- MODEL TRAINING ----------------

def train_clustering_model(images, clusters, epochs):

    import tensorflow as tf

    tf.keras.backend.clear_session()

    input_img = tf.keras.layers.Input(
        shape=(32, 32, 3)
    )

    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    )(input_img)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2), padding="same"
    )(x)

    x = tf.keras.layers.Conv2D(
        16, (3, 3), activation="relu", padding="same"
    )(x)

    encoded = tf.keras.layers.MaxPooling2D(
        (2, 2), padding="same"
    )(x)

    x = tf.keras.layers.Conv2D(
        16, (3, 3), activation="relu", padding="same"
    )(encoded)

    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    )(x)

    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    decoded = tf.keras.layers.Conv2D(
        3, (3, 3), activation="sigmoid", padding="same"
    )(x)

    autoencoder = tf.keras.Model(input_img, decoded)

    encoder = tf.keras.Model(input_img, encoded)

    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    history = autoencoder.fit(
        images,
        images,
        epochs=epochs,
        batch_size=min(32, len(images)),
        shuffle=True,
        verbose=0
    )

    features = encoder.predict(
        images, verbose=0
    )

    features = features.reshape(
        features.shape[0], -1
    )

    kmeans = KMeans(
        n_clusters=clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(features)

    return encoder, kmeans, labels, history.history["loss"]


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🚦 VISION-CLUSTER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Traffic Sign Image Analysis & Machine Learning Clustering'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- LOAD DATA ----------------

images, image_names = load_dataset()

total_images = len(images)


# ---------------- SIDEBAR ----------------

st.sidebar.title("🚦 Vision Cluster")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Dataset Explorer",
        "Train ML Model",
        "Image Clustering",
        "Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "CNN Autoencoder + K-Means Clustering"
)


# ---------------- OVERVIEW ----------------

if page == "Dashboard Overview":

    st.subheader("📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Images", total_images)

    col2.metric(
        "Model Status",
        "Trained" if "model_bundle" in st.session_state
        else "Not Trained"
    )

    col3.metric(
        "Algorithm",
        "K-Means"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("About the Project")

        st.write("""
        Vision-Cluster is an unsupervised Machine Learning
        project designed to group similar traffic sign images.

        It uses:

        - CNN Autoencoder for feature extraction
        - K-Means for image clustering
        - Streamlit for interactive visualization
        """)

    with right:
        st.subheader("Project Workflow")

        st.markdown("""
        1. Load traffic sign images.
        2. Resize and normalize images.
        3. Extract features using CNN.
        4. Apply K-Means clustering.
        5. Visualize cluster results.
        6. Analyze uploaded images.
        """)

    if total_images == 0:
        st.warning(
            "No images found. Add image files inside the dataset folder."
        )

    st.divider()

    st.subheader("Dashboard Features")

    c1, c2, c3 = st.columns(3)

    c1.info("📁 Dataset Analysis")
    c2.info("🧠 Model Training")
    c3.info("📈 Cluster Visualization")


# ---------------- DATASET EXPLORER ----------------

elif page == "Dataset Explorer":

    st.subheader("📁 Dataset Explorer")

    st.metric("Total Images Available", total_images)

    if total_images == 0:
        st.warning("No valid images found in dataset/")
    else:

        df = pd.DataFrame({
            "Image": image_names
        })

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Image Preview")

        preview_count = min(12, total_images)

        cols = st.columns(4)

        for i in range(preview_count):
            with cols[i % 4]:
                st.image(
                    images[i],
                    caption=image_names[i],
                    use_container_width=True
                )


# ---------------- MODEL TRAINING ----------------

elif page == "Train ML Model":

    st.subheader("🧠 Train Clustering Model")

    st.write(
        "Train the CNN Autoencoder and K-Means model "
        "using the images in your dataset."
    )

    cluster_count = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=min(10, max(2, total_images)),
        value=min(5, max(2, total_images))
    )

    epochs = st.slider(
        "Training Epochs",
        min_value=1,
        max_value=30,
        value=5
    )

    st.write("Available training images:", total_images)

    if st.button("🚀 Start Model Training", type="primary"):

        if total_images < 2:
            st.error("At least two valid images are required.")

        elif cluster_count > total_images:
            st.error("Clusters cannot exceed image count.")

        else:
            try:
                with st.spinner("Training CNN and K-Means..."):

                    encoder, kmeans, labels, losses = (
                        train_clustering_model(
                            images,
                            cluster_count,
                            epochs
                        )
                    )

                    st.session_state.model_bundle = {
                        "encoder": encoder,
                        "kmeans": kmeans,
                        "labels": labels,
                        "losses": losses,
                        "clusters": cluster_count
                    }

                st.success("Model training completed!")

                st.metric(
                    "Final Reconstruction Loss",
                    f"{losses[-1]:.5f}"
                )

                loss_fig = go.Figure()

                loss_fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(losses) + 1)),
                        y=losses,
                        mode="lines+markers",
                        name="Training Loss"
                    )
                )

                loss_fig.update_layout(
                    title="Autoencoder Training Loss",
                    xaxis_title="Epoch",
                    yaxis_title="MSE Loss",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    loss_fig,
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Training failed: {e}")


# ---------------- IMAGE CLUSTERING ----------------

elif page == "Image Clustering":

    st.subheader("🖼️ Upload Traffic Sign Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file:

        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                img,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            if "model_bundle" not in st.session_state:
                st.warning(
                    "Train the model first using the Train ML Model page."
                )

            else:
                if st.button("Analyze Image"):

                    try:
                        bundle = st.session_state.model_bundle

                        processed = img.resize(IMAGE_SIZE)

                        arr = (
                            np.asarray(processed, dtype=np.float32)
                            / 255.0
                        )

                        arr = np.expand_dims(arr, axis=0)

                        features = bundle["encoder"].predict(
                            arr,
                            verbose=0
                        )

                        features = features.reshape(1, -1)

                        cluster = bundle["kmeans"].predict(
                            features
                        )[0]

                        st.success("Analysis completed!")

                        st.metric(
                            "Assigned Cluster",
                            f"Cluster {cluster}"
                        )

                        st.info(
                            "This is an unsupervised cluster assignment, "
                            "not a verified traffic sign classification."
                        )

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")


# ---------------- CLUSTER ANALYSIS ----------------

    if "model_bundle" in st.session_state:

        st.divider()

        st.subheader("📊 Cluster Distribution")

        bundle = st.session_state.model_bundle

        df = pd.DataFrame({
            "Image": image_names,
            "Cluster": bundle["labels"]
        })

        counts = (
            df["Cluster"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        counts.columns = ["Cluster", "Images"]

        fig = px.bar(
            counts,
            x="Cluster",
            y="Images",
            color="Cluster",
            title="Images in Each Cluster",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Cluster Results",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="vision_cluster_results.csv",
            mime="text/csv"
        )


# ---------------- MODEL INFORMATION ----------------

elif page == "Model Information":

    st.subheader("⚙️ Machine Learning Model")

    st.markdown("""
    **1. CNN Autoencoder**

    Extracts compact visual features from traffic sign images.

    **2. K-Means Clustering**

    Groups images according to similarities in their extracted
    feature representations.

    **3. Streamlit Dashboard**

    Provides an interactive interface for dataset exploration,
    training, image uploads and cluster analysis.
    """)

    st.divider()

    st.subheader("Model Parameters")

    st.write("- Image dimensions: 32 × 32 × 3")
    st.write("- Optimizer: Adam")
    st.write("- Loss function: Mean Squared Error")
    st.write("- Clustering algorithm: K-Means")
    st.write("- Random state: 42")

    if "model_bundle" in st.session_state:

        bundle = st.session_state.model_bundle

        st.success("A trained model is available in this session.")

        st.write(
            "Number of clusters:",
            bundle["clusters"]
        )

    else:
        st.warning(
            "No model trained in this session yet."
        )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Vision-Cluster | Machine Learning | "
    "CNN Autoencoder + K-Means | Streamlit"
)
