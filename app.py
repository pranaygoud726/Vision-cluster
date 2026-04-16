from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Read image
            img = cv2.imread(filepath)
            img_resized = cv2.resize(img, (64, 64))
            img_flat = img_resized.flatten().reshape(1, -1)

            # Dummy clustering
            kmeans = KMeans(n_clusters=5, random_state=0)
            kmeans.fit(np.random.rand(10, img_flat.shape[1]))

            cluster = kmeans.predict(img_flat)[0]

            image_url = url_for('static', filename='uploads/' + file.filename)

            return render_template('index.html', image=image_url, cluster=cluster)

    return render_template('index.html', image=None)

# 🔥 Render deployment fix
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
