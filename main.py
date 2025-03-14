from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    try:
        # Get JSON data
        data = request.get_json()

        # Debugging - print received data
        print("Received data:", data)
        print("Data type:", type(data))

        # Check if the data is a string instead of a dictionary
        if isinstance(data, str):  
            data = json.loads(data)  # Convert to dictionary
        
        # Access values safely
        if "key" not in data:
            return jsonify({"error": "Missing 'key' in request"}), 400
        
        value = data["key"]  # Extract key

        # Process the data (Modify this based on your needs)
        processed_value = f"Processed: {value}"

        return jsonify({"result": processed_value})

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
