from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

# Hugging Face Gradio API Client
client = Client("pratyyush/image-colorizer-deoldify_working")

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

        # Extract colorized image URL
        colorized_image_url = result if isinstance(result, str) else result.get("output", "")

        if not colorized_image_url:
            return jsonify({"error": "No output image received from Hugging Face"}), 500

        return jsonify({"colorized_image_url": colorized_image_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
