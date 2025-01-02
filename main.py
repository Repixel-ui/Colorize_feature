from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import os
import requests

app = Flask(__name__)

# Directories for the model files
MODEL_DIR = "models"
PROTOTXT = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
POINTS = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Download files from Google Drive
def download_from_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            f.write(response.content)
    else:
        print(f"Error downloading file {file_id}")

# Initialize the model
def initialize_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(PROTOTXT):
        download_from_drive("1hK56NLhwHxI61Zn3rSs7oJAu3KfCa_nI", PROTOTXT)
    if not os.path.exists(MODEL):
        download_from_drive("1Q8-iJjv4I7VfqTTr4VjNUpUi8ZtaagFm", MODEL)
    if not os.path.exists(POINTS):
        download_from_drive("1evjjUeX3PN0pz0qX2Q8lZmkFq2khdjsj", POINTS)

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    return net

# Load the model
net = None
@app.before_first_request
def load_model():
    global net
    initialize_model()
    net = initialize_model()

@app.route("/colorize", methods=["POST"])
def colorize_image():
    # Check if an image is sent
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Save the uploaded image
    file = request.files["image"]
    image_path = "input_image.jpg"
    file.save(image_path)

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorize the image
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Save and send the colorized image
    output_path = "colorized_image.jpg"
    cv2.imwrite(output_path, colorized)
    return send_file(output_path, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
