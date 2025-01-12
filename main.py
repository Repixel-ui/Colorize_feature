from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import os
import gdown
from flask_cors import CORS
import gc

app = Flask(__name__)
CORS(app)

# Define directories for models and image paths
DIR = "./models"
PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "pts_in_hull.npy")
MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")

# URLs to download large files dynamically
MODEL_URL = "https://drive.google.com/uc?id=1Q8-iJjv4I7VfqTTr4VjNUpUi8ZtaagFm&export=download"
POINTS_URL = "https://drive.google.com/uc?id=1evjjUeX3PN0pz0qX2Q8lZmkFq2khdjsj&export=download"
PROTOTXT_URL = "https://drive.google.com/uc?id=1hK56NLhwHxI61Zn3rSs7oJAu3KfCa_nI&export=download"

# Function to download files if not present
def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        try:
            gdown.download(url, filepath, quiet=False)
            print(f"Saved {filepath}")
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False
    return True

# Ensure model files are available
os.makedirs(DIR, exist_ok=True)
download_file(MODEL_URL, MODEL)
download_file(POINTS_URL, POINTS)
download_file(PROTOTXT_URL, PROTOTXT)

# Load the model only once at the beginning
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
if net.empty():
    print("Error loading the model")
    exit()

print("Model loaded successfully")

# Load centers for ab channel quantization
try:
    pts = np.load(POINTS)
    print("Points loaded successfully")
except Exception as e:
    print(f"Error loading points: {e}")
    exit()

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

@app.route("/colorize", methods=["POST"])
def colorize():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    input_path = os.path.join(DIR, "input_image.jpg")
    output_path = os.path.join(DIR, "colorized_output.jpg")
    file.save(input_path)

    # Load the input image
    image = cv2.imread(input_path)
    if image is None:
        return jsonify({"error": "Unable to load image"}), 400

    # Resize the input image to a smaller size to reduce memory consumption
    small_size = (300, 300)
    image_resized = cv2.resize(image, small_size)

    # Preprocess the resized image (scale and convert to LAB)
    scaled = image_resized.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize and extract the luminance channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorize the image
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the ab channels back to the resized image size
    ab = cv2.resize(ab, (image_resized.shape[1], image_resized.shape[0]))

    # Concatenate the channels to form a colorized image
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert from LAB to BGR color space
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Scale back to [0, 255] and convert to uint8
    colorized = (255 * colorized).astype("uint8")

    # Save the colorized image
    cv2.imwrite(output_path, colorized)

    # Clear memory and remove large variables after use
    del image, image_resized, scaled, lab, resized, L, ab, colorized
    gc.collect()

    return send_file(output_path, mimetype="image/jpeg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
