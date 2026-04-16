from flask import Flask, render_template, request
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Simple processing (dummy cluster for demo)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (32,32))
        img = img / 255.0

        # Fake cluster output (you can connect model later)
        cluster = np.random.randint(0,5)

        return render_template("index.html", image=filepath, cluster=cluster)

    return render_template("index.html", image=None)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    
