from flask import Flask, request, jsonify
import os
import requests
from gradio_client import Client

app = Flask(__name__)

# Hugging Face Spaces URL where DeOldify is deployed
HUGGING_FACE_API_URL = "https://your-deoldify-space.hf.space/"

@app.route("/", methods=["GET"])
def home():
    return "Flask server is running on Render!"

@app.route("/colorize", methods=["POST"])
def colorize_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = "temp.jpg"
    image.save(image_path)

    try:
        client = Client(HUGGING_FACE_API_URL)
        result = client.predict(image_path, api_name="/predict")

        return jsonify({"colorized_url": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
