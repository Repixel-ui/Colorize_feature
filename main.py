from flask import Flask, request, jsonify
from gradio_client import Client, handle_file

app = Flask(__name__)

client = Client("pratyyush/image-colorizer-deoldify_working")

@app.route("/colorize", methods=["POST"])
def colorize():
    if "image" in request.files:  # If an image file is sent
        image = request.files["image"]
        image_path = f"./{image.filename}"  # Save temporarily
        image.save(image_path)  

        result = client.predict(
            image_path=handle_file(image_path),
            api_name="/predict"
        )
        
        return jsonify({"colorized_image": result})
    
    elif request.is_json:  # If JSON is sent
        data = request.get_json()
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "Missing 'image_url'"}), 400

        result = client.predict(
            image_path=handle_file(image_url),
            api_name="/predict"
        )

        return jsonify({"colorized_image": result})

    else:
        return jsonify({"error": "Invalid request. Send an image file or JSON with 'image_url'."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
