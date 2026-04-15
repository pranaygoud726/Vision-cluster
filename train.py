import cv2
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.cluster import KMeans

# Load dataset
data = []
path = "dataset/"

for root, dirs, files in os.walk(path):
    for file in files:
        img_path = os.path.join(root, file)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, (32, 32))
            img = img / 255.0
            data.append(img)

data = np.array(data)

print("Total images:", len(data))

# Autoencoder
input_img = Input(shape=(32, 32, 3))

x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(data, data, epochs=10, batch_size=32)

# Extract features
encoder = Model(input_img, encoded)
features = encoder.predict(data)
features = features.reshape(features.shape[0], -1)

# KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(features)

print("Clustering completed!")