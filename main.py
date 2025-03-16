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

        print(f"✅ Image saved locally: {temp_filename}")
        return temp_filename
    except Exception as e:
        print("❌ Error decoding Base64:", str(e))
        return None

def upload_to_imgur(image_path):
    """Uploads an image to Imgur and returns the public URL."""
    try:
        with open(image_path, "rb") as image_file:
            img_data = {"image": image_file}
            headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
            response = requests.post("https://api.imgur.com/3/image", headers=headers, files=img_data)

        if response.status_code == 200:
            imgur_url = response.json()["data"]["link"]
            print(f"✅ Image uploaded to Imgur: {imgur_url}")
            return imgur_url
        else:
            print(f"❌ Imgur upload failed: {response.json()}")
            return None
    except Exception as e:
        print("❌ Error uploading to Imgur:", str(e))
        return None

@app.route('/colorize', methods=['POST'])
def colorize_image():
    try:
        # Step 1: Get Base64 image from request
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        base64_image = fix_base64_padding(data["image"])  # Fix padding
        image_path = save_base64_image(base64_image)  # Save locally

        if not image_path:
            return jsonify({"error": "Failed to process image"}), 500

        # Step 2: Upload input image to Imgur first (OPTIONAL, if needed)
        # Comment this out if Hugging Face supports direct file input
        # input_imgur_url = upload_to_imgur(image_path)
        # if not input_imgur_url:
        #     print("❌ Failed to upload input image to Imgur")
        #     return jsonify({"error": "Failed to upload input image"}), 500

        # Step 3: Send image to Hugging Face API
        print("🔄 Sending image to Hugging Face for colorization...")
        result = client.predict(
            image_path,  # Send local image path (or input_imgur_url if needed)
            api_name="/predict"
        )

        # 🛑 DEBUG: Print the Hugging Face response
        print(f"🛑 Hugging Face Model Response: {result}")

        if not isinstance(result, str) or not os.path.exists(result):
            print(f"❌ Invalid Hugging Face response: {result}")
            return jsonify({"error": "Invalid response from Hugging Face"}), 500

        colorized_image_path = result  # Colorized image path

        # Step 4: Check if the colorized image exists
        if not os.path.exists(colorized_image_path):
            print(f"❌ Colorized image not found at {colorized_image_path}")
            return jsonify({"error": "Colorized image not found"}), 500

        print(f"✅ Colorized image saved: {colorized_image_path}")

        # Step 5: Upload colorized image to Imgur
        print("🔄 Uploading colorized image to Imgur...")
        imgur_url = upload_to_imgur(colorized_image_path)

        if not imgur_url:
            print("❌ Failed to upload to Imgur")
            return jsonify({"error": "Failed to upload to Imgur"}), 500

        return jsonify({"colorized_image_url": imgur_url})

    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        print(f"❌ Full Error:\n{error_message}")  # Print full error details
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
