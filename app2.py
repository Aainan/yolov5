import os
from flask import Flask, jsonify, request, send_file
import torch
from PIL import Image
import io

app = Flask(__name__)

# Construct the absolute path to the model
model_path = os.path.abspath('/best.pt')

# Load your YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Open the image file
    image = Image.open(file.stream)
    
    # Perform inference
    results = model(image)
    
    # Save the result image
    result_img = results.render()[0]
    img_bytes = io.BytesIO()
    result_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Return the result image
    return send_file(img_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
