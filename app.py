import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- 1. LOAD THE AI MODELS ---
print("🧠 Loading StyleVision AI Models...")
clothing_model = tf.keras.models.load_model('stylevision_v1.h5')
class_names = np.load('class_names.npy', allow_pickle=True)
yolo_detector = YOLO('yolov8n.pt') 

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_image_pipeline(image_path):
    """The High-Performance Hybrid Pipeline"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # STAGE 1: YOLO GATEKEEPER
    results = yolo_detector(img_rgb, verbose=False)
    boxes = results[0].boxes
    
    # Things we KNOW are not clothes (Bottle, Cup, Chair, Clock)
    REJECT_CLASSES = [39, 41, 62, 73] 
    target_img = img_rgb # Default to full image if no crop found

    if len(boxes) > 0:
        top_box = boxes[0]
        class_id = int(top_box.cls[0])
        conf = float(top_box.conf[0])

        # Hard Reject for known non-apparel
        if class_id in REJECT_CLASSES and conf > 0.4:
            return "Non-Apparel Item (Rejected by YOLO)", conf

        # If YOLO finds a 'person' or 'handbag', crop to that area
        x1, y1, x2, y2 = map(int, top_box.xyxy[0])
        target_img = img_rgb[y1:y2, x1:x2]

    # STAGE 2: CLASSIFICATION
    resized_img = cv2.resize(target_img, (224, 224))
    input_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.array([resized_img], dtype=np.float32)
    )
    
    predictions = clothing_model.predict(input_tensor, verbose=0)[0]
    max_index = np.argmax(predictions)
    confidence = float(predictions[max_index])
    
    # STAGE 3: SMART THRESHOLDING
    # We require 85% confidence for full images, 70% for YOLO crops
    required_conf = 0.70 if len(boxes) > 0 else 0.85

    if confidence < required_conf:
        return "Unknown / Low Confidence", confidence
        
    return class_names[max_index], confidence

# --- FLASK ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file submission'})

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        predicted_class, confidence = process_image_pipeline(filepath)
        
        return jsonify({
            'prediction': str(predicted_class),
            'confidence': f"{confidence * 100:.2f}%"
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)