from flask import Flask, request, jsonify
import requests
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('fruition_model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img_path = 'temp.jpg'  # Temporary file to save the uploaded image
        file.save(img_path)

        processed_img = preprocess_image(img_path)
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions)

        classes = ['Apple Fresh', 'Apple Rotten', 'Apple Semifresh', 'Apple Semirotten',
                   'Banana Fresh', 'Banana Rotten', 'Banana Semifresh', 'Banana Semirotten',
                   'Orange Fresh', 'Orange Rotten', 'Orange Semifresh', 'Orange Semirotten']

        predicted_label = classes[predicted_class]
        accuracy = float(predictions[0][predicted_class] * 100.0)

        # Assuming you have access to the user's token in your Flask app
        user_token = request.headers.get('Authorization').split('Bearer ')[1]

        # Send user's token in the header to Express
        express_url = 'http://localhost:3000/histories/'  # Replace with your Express URL
        payload = {
            'predicted_class': predicted_label,
            'prediction_accuracy': accuracy
        }
        headers = {
            'content-type': 'application/json',
            'Authorization': f'{user_token}'  # Include the user's token in the header
        }
        response = requests.post(express_url, data=json.dumps(payload), headers=headers)

        return jsonify({
            'fruit': predicted_label,
            'prediction_accuracy': accuracy,
            'message_sent_to_express': response.text
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
