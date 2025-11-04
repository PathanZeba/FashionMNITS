from flask import Flask, request, jsonify, render_template
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "models/fashion_mnist_cnn.keras"
model = keras.models.load_model(MODEL_PATH)

# Class labels for Fashion-MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Check if file is in request
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Check if filename is empty
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        # Preprocess uploaded image
        img = Image.open(file.stream).convert("L").resize((28, 28))
        img = ImageOps.invert(img)   # Invert colors to match Fashion-MNIST style
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=-1)  # shape: (28,28,1)
        arr = np.expand_dims(arr, axis=0)   # shape: (1,28,28,1)

        # Predict
        probs = model.predict(arr)[0]
        idx = int(np.argmax(probs))

        return jsonify({
            "pred_index": idx,
            "pred_label": CLASS_NAMES[idx],
            "probability": float(np.max(probs))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == "__main__":
    app.run(debug=False)