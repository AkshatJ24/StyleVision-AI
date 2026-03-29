# 🧠 StyleVision AI: Two-Stage Apparel Intelligence

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Framework-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8_Detection-00FFFF?logo=ultralytics)](https://ultralytics.com/)

StyleVision AI is a production-grade computer vision pipeline designed to solve the "naive classifier" problem. While standard models forcedly categorize every input, StyleVision uses a multi-stage architecture to detect, crop, and validate apparel before making a final classification, presented through a sleek, interactive web interface.

> **🎥 Note to Recruiters/Reviewers:** Check out the `/samples` folder for demo photos showcasing the pipeline in action.

## 🚀 Key Features
- **Two-Stage Pipeline:** Integrates YOLOv8 for object localization and MobileNetV2 for fine-grained classification.
- **Dynamic Cropping:** Automatically extracts the Region of Interest (ROI) to improve accuracy and ignore background noise.
- **Intelligent Filtering:** Prevents False Positives through custom YOLO class-filtering and dual-confidence thresholds (e.g., actively rejects non-apparel items like cups or furniture).
- **Interactive UI:** A modern, dark-themed web interface with real-time prediction feedback and dynamic category highlighting.

## 🏗️ The Architecture
1. **Detection (YOLOv8):** The "Gatekeeper" layer. It scans the image for persons or apparel-related items. If a non-apparel item is detected with high confidence, the system rejects it immediately.
2. **Classification (MobileNetV2):** The "Brain" layer. The cropped image is passed to a Transfer Learning model fine-tuned on 15,000+ high-resolution fashion images.
3. **The Guardrail:** A "Smart Threshold" system requires 85% confidence for full-image guesses and 70% for cropped detections, ensuring high reliability.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow / Keras (MobileNetV2)
- **Object Detection:** Ultralytics (YOLOv8)
- **Computer Vision:** OpenCV
- **Backend:** Python, Flask
- **Frontend:** HTML5, Modern CSS, Vanilla JS

## 👔 Supported Categories
The model is trained to recognize 10 distinct apparel and accessory classes:
- **Clothing:** Tshirts, Shirts, Tops, Kurtas, Dresses
- **Footwear:** Casual Shoes, Heels
- **Accessories:** Handbags, Watches, Sunglasses

## 📂 Project Structure
```text
StyleVision-AI/
├── samples/                  # Demo videos, before/after images, and edge-cases
├── static/
│   └── uploads/              # Temporary storage for image processing
├── templates/
│   └── index.html            # Interactive web interface
├── app.py                    # Flask application and hybrid inference pipeline
├── class_names.npy           # Encoded category labels
├── requirements.txt          # Python dependencies
├── stylevision_v1.h5         # Fine-tuned MobileNetV2 weights
└── yolov8n.pt                # YOLOv8 nano weights
```

## 🏃 Local Setup
Want to run this pipeline on your own machine? Follow these steps:

**1. Clone the Repository**
```bash
git clone https://github.com/AkshatJ24/StyleVision-AI.git
```
```
cd StyleVision-AI
```

**2. Create a Virtual Environment (Recommended)**
```
python -m venv .venv
# On Windows use: 
.venv\Scripts\activate
# On macOS/Linux use: 
source .venv/bin/activate
```

**3. Install Dependencies**
```
pip install -r requirements.txt
```

**4. Run the Application**
```
python app.py
```

## 📚 Syllabus Relevance
This project directly applies concepts from **Module 3: Feature Extraction and Image Segmentation**. By utilizing YOLOv8 for dynamic region-of-interest cropping (advanced segmentation) and MobileNetV2 for deep feature extraction, the pipeline demonstrates a modern, practical application of core computer vision principles.