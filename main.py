import requests
import base64
from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Hugging Face Gradio API Client
client = Client("pratyyush/image-colorizer-deoldify_working")

# Imgur API Client ID (replace with your own)
IMGUR_CLIENT_ID = "8d58d2a8c959a1b"

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
        # Get base64 image from Android
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        base64_image = data["image"]

        # Send image to Hugging Face API
        result = client.predict(
            base64_image,  # Sending Base64 directly
            api_name="/predict"
        )

        # Extract local path of the colorized image
        if isinstance(result, str):
            colorized_image_path = result
        else:
            colorized_image_path = result.get("colorized_image", "")

        if not colorized_image_path:
            return jsonify({"error": "No output image received from Hugging Face"}), 500

        # Upload colorized image to Imgur
        imgur_url = upload_to_imgur(colorized_image_path)
        if not imgur_url:
            return jsonify({"error": "Failed to upload to Imgur"}), 500

        return jsonify({"colorized_image_url": imgur_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
