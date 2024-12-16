import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model = load_model('model.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Define labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess image and get prediction
def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

# Route for the homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify('No file part'), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify('No selected file'), 400

    if f:
        filename = secure_filename(f.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(file_path)
        
        # Get the prediction
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        
        # Optional: Remove the file after prediction
        os.remove(file_path)
        
        return jsonify(predicted_label)

    return jsonify('Error during file upload'), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
