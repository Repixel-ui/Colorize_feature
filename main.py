import base64
import os
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file

app = Flask(__name__)
client = Client("pratyyush/image-colorizer-deoldify_working")

@app.route('/colorize', methods=['POST'])
def colorize_image():
    try:
        # Get base64 image from request
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        base64_image = data["image"]

        # Decode Base64 and save as a temporary file
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as img_file:
            img_file.write(base64.b64decode(base64_image))

        # Send image to Hugging Face API
        result = client.predict(
            handle_file(image_path),  # Use handle_file to send file
            api_name="/predict"
        )

        # Clean up temp file
        os.remove(image_path)

        return jsonify({"colorized_image_url": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
