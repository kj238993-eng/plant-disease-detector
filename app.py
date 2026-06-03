import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limit to 10MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuration & mappings
IMG_SIZE = (128, 128)  # Must match the training input size
MODEL_PATH = 'plant_model.h5'

CLASS_KEY_MAPPING = {
    'Potato___Early_blight': 'early',
    'Potato___Late_blight': 'late',
    'Potato___healthy': 'healthy'
}

DISPLAY_NAMES = {
    'Potato___Early_blight': 'Early Blight',
    'Potato___Late_blight': 'Late Blight',
    'Potato___healthy': 'Healthy'
}

# Load model globally at startup
print("Loading TensorFlow Keras CNN model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first by running train.py.")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load class index mapping
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    # class_indices maps "0" -> "Potato___Early_blight", etc.
    # Keras returns indexes which we map using this json
    idx_to_class = {int(k): v for k, v in class_indices.items()}

print("Class indices loaded:", idx_to_class)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'webp', 'bmp'}

def preprocess_image(image_path):
    # Load image and force RGB (handling PNG transparency/RGBA)
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    # Convert to array, rescale is handled inside the model's rescaling layer
    arr = np.array(img, dtype=np.float32)
    # Add batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File format not supported. Upload JPG, PNG or BMP.'}), 400
    
    # Save the file safely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Preprocess and predict
        arr = preprocess_image(filepath)
        preds = model.predict(arr, verbose=0)[0]  # Shape: (3,)
        
        # Determine classification
        pred_idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100
        class_name = idx_to_class[pred_idx]
        
        # Mappings for frontend
        class_key = CLASS_KEY_MAPPING[class_name]
        
        # Map probabilities for all classes
        all_probs = {
            DISPLAY_NAMES[idx_to_class[i]]: round(float(preds[i]) * 100, 1)
            for i in range(len(preds))
        }
        
        return jsonify({
            'classKey': class_key,
            'confidence': round(confidence, 1),
            'all_probs': all_probs
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Model inference error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask web server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
