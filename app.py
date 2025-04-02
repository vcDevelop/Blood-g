import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Ensure required libraries are installed
try:
    import flask
    import tensorflow
    import numpy
except ImportError as e:
    print("Missing dependencies. Installing...")
    os.system("pip install flask tensorflow numpy")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (Ensure the model file is available)
MODEL_PATH = "blood_group_vgg16_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please download it before running the app.")

model = load_model(MODEL_PATH)

# Blood group labels
blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# Upload folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        # Save file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Predict blood group
        prediction = predict_blood_group(filepath)
        return jsonify({"prediction": prediction})

    return render_template("index.html")

def predict_blood_group(img_path):
    """ Preprocesses the image and predicts the blood group """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return blood_groups[predicted_class]
    except Exception as e:
        return f"Error processing image: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)