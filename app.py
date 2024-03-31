from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import requests
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS

# Define image size and color mode
img_height, img_width, color_mode = 224, 224, 'grayscale'

# Load model and preprocess image
def load_and_preprocess_image(model_path, image_data):
    model = load_model(model_path)
    img = img_to_array(load_img(BytesIO(image_data), target_size=(img_height, img_width), color_mode=color_mode))
    img_batch = np.expand_dims(img, axis=0)
    return model, img_batch

# Define label mapping
label_mapping = {0: 'Bacterial', 1: 'Normal', 2: 'Viral'}

# Define Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/upload', methods=['POST'])
def classify_image():
    try:
        # Get the URL string of the image from the request
        url = request.json.get('url')
        if not url:
            return jsonify({'error': 'No URL received'})

        # Fetch image from URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        image_data = response.content  # Image data retrieved from URL

        # Load and preprocess image
        model, img_batch = load_and_preprocess_image('model4.h5', image_data)

        # Perform inference using the loaded model
        predictions = model.predict(img_batch)
        predicted_label_index = np.argmax(predictions[0])
        predicted_label = label_mapping[predicted_label_index]
        confidence = float(np.max(predictions) * 100)  # Convert to float and percentage

        # Round off confidence to two decimal places
        confidence_rounded = round(confidence, 2)

        result = {
            'predicted_label': predicted_label,
            'confidence': confidence_rounded
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
