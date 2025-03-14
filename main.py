import os
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file

app = Flask(__name__)

# Load Gradio client
client = Client("pratyyush/image-colorizer-deoldify_working")

@app.route("/", methods=["GET"])
def home():
    return "Flask server is running!"

@app.route("/colorize", methods=["POST"])
def colorize():
    try:
        # Get image URL from request
        data = request.get_json()
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "Missing image_url"}), 400

        # Send to Gradio API
        result = client.predict(
            image_path=handle_file(image_url),
            api_name="/predict"
        )

        return jsonify({"colorized_image": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render needs to use PORT
    app.run(host="0.0.0.0", port=port, debug=True)
