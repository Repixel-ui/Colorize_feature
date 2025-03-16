import requests
import base64
import os
from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Hugging Face DeOldify Model API
client = Client("pratyyush/image-colorizer-deoldify_working")

# Imgur API Client ID (Replace with your own Client ID)
IMGUR_CLIENT_ID = "8d58d2a8c959a1b"

def fix_base64_padding(base64_string):
    """Fix Base64 padding issues by adding missing '=' characters."""
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += "=" * (4 - missing_padding)
    return base64_string

def save_base64_image(base64_str):
    """Decode and save Base64 image to a temporary file."""
    try:
        image_data = base64.b64decode(base64_str)
        temp_filename = "input_image.jpg"
        with open(temp_filename, "wb") as f:
            f.write(image_data)
        return temp_filename
    except Exception as e:
        print("Error decoding Base64:", str(e))
        return None

def upload_to_imgur(image_path):
    """Uploads an image to Imgur and returns the public URL."""
    with open(image_path, "rb") as image_file:
        img_data = {"image": image_file}
        headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
        response = requests.post("https://api.imgur.com/3/image", headers=headers, files=img_data)

    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        return None

@app.route('/colorize', methods=['POST'])
def colorize_image():
    try:
        # Get Base64 image from request
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        base64_image = fix_base64_padding(data["image"])  # Fix padding
        image_path = save_base64_image(base64_image)  # Save locally

        if not image_path:
            return jsonify({"error": "Failed to process image"}), 500

        # Send image to Hugging Face API
        result = client.predict(
            image_path,  # Send local image path
            api_name="/predict"
        )

        if not isinstance(result, str):
            return jsonify({"error": "Invalid response from Hugging Face"}), 500

        colorized_image_path = result  # Colorized image path

        # Upload colorized image to Imgur
        imgur_url = upload_to_imgur(colorized_image_path)
        if not imgur_url:
            return jsonify({"error": "Failed to upload to Imgur"}), 500

        return jsonify({"colorized_image_url": imgur_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
