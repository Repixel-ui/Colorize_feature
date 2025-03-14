from flask import Flask, request, jsonify, send_from_directory
from gradio_client import Client, handle_file
import os
import shutil

app = Flask(__name__)

# Initialize Hugging Face Gradio Client
gradio_client = Client("pratyyush/image-colorizer-deoldify_working")

# Create a static directory if it doesn't exist
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/colorize', methods=['POST'])
def colorize_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']
    image_path = os.path.join(STATIC_FOLDER, image_file.filename)
    image_file.save(image_path)  # Save uploaded file locally

    # Send the image to Hugging Face Gradio API
    result = gradio_client.predict(
        image_path=handle_file(image_path),
        api_name="/predict"
    )

    colorized_image_path = result["colorized_image"]  # Get local processed image path

    # Move processed image to static folder
    filename = os.path.basename(colorized_image_path)
    new_image_path = os.path.join(STATIC_FOLDER, filename)
    shutil.move(colorized_image_path, new_image_path)

    # Generate a public URL for the image
    public_url = f"https://colorize-feature.onrender.com/static/{filename}"

    return jsonify({"colorized_image_url": public_url})

# Route to serve images from /static folder
@app.route('/static/<filename>')
def serve_image(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
