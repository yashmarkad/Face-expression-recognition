from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to 48x48
    image = image.resize((48, 48))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image to the range [0, 1]
    image = image / 255.0
    # Reshape the image to match the model's input shape
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the image file
        image = Image.open(io.BytesIO(file.read()))
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Make a prediction
        predictions = model.predict(processed_image)
        # Get the predicted emotion
        predicted_emotion = emotion_labels[np.argmax(predictions)]
        # Return the result as JSON
        return jsonify({'emotion': predicted_emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)