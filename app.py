from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


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
        if image_file.filename != '':
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            image = cv2.imread(image_location)
            image = cv2.resize(image, (256, 256))
            image = image.astype('float32')
            image /= 255.0
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)
            prediction = prediction.flatten()   # Flatten the array
            depth_map = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255
            depth_map = depth_map.reshape((256, 256))  # Reshape the array to 2D
            prediction = prediction.tolist()   # Convert the array to a list after normalization
            # Save the depth map to a file
            depth_map_path = os.path.join('static', 'depth_map.png')
            plt.imsave(depth_map_path, depth_map, cmap='gray')
            # Save the depth values to a text file
            depth_values_path = os.path.join('static', 'depth_values.txt')
            np.savetxt(depth_values_path, depth_map)
            return render_template('index.html', prediction=depth_map_path)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(port=5000, debug=True)