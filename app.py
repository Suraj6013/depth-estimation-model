from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your model
model = load_model(r'C:\Users\suraj\OneDrive\Desktop\Projects\depth estimation model\my_model')

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            image = cv2.imread(image_location)
            image = cv2.resize(image, (256, 256))  # Update this line
            image = image.astype('float32')
            image /= 255.0
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)
            # You can process the prediction result here
            return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(port=5000, debug=True)