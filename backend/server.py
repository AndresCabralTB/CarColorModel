from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained YOLOv8 model
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the model
model_path = os.path.join(current_dir, 'models', 'best_v2.pt')

# Load the model
model = YOLO(model_path)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML upload form

# Route for image upload and detection
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        # Read the uploaded image
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Ensure RGB format

        # Perform detection
        results = model.predict(source=pil_image, save=False, imgsz=640)
        result = results[0]

        # Annotate the image with detections
        annotated_image = result.plot()

        # Convert BGR (default OpenCV) to RGB for proper display in browsers
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Encode the annotated image as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_image)

        return send_file(
            io.BytesIO(buffer.tobytes()),
            mimetype='image/jpeg',
            as_attachment=False,
            download_name='detected.jpg'
        )
    except Exception as e:
        print(f"Error during processing: {e}")
        return "Error processing the image", 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
