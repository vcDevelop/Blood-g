import os
import numpy as np
import tensorflow as tf
import gdown  # Import gdown to download files from Google Drive
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Google Drive file ID (Extract from shareable link)
FILE_ID = "1g7ShOzQjH_ueUv2RT_B-pcerAifNd25A"
MODEL_PATH = "blood_group_vgg16_model.h5"

# Download model if not available locally
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# Load the trained model
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
