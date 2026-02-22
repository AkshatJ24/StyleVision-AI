# 🧠 StyleVision AI: Two-Stage Apparel Intelligence

StyleVision AI is a production-grade computer vision pipeline designed to solve the "naive classifier" problem. While standard models forcedly categorize every input, StyleVision uses a multi-stage architecture to detect, crop, and validate apparel before making a final classification.

## 🚀 Key Features
- **Two-Stage Pipeline:** Integrates YOLOv8 for object localization and MobileNetV2 for fine-grained classification.
- **Dynamic Cropping:** Automatically extracts the Region of Interest (ROI) to improve accuracy and ignore background noise.
- **Intelligent Filtering:** Prevents False Positives through custom YOLO class-filtering and dual-confidence thresholds.
- **Robust Evaluation:** Achieved 94.5% accuracy across 10 diverse fashion categories.

## 🏗️ The Architecture
1. **Detection (YOLOv8):** The "Bouncer" layer. It scans the image for persons or apparel-related items (Handbags, Backpacks, etc.). If a non-apparel item like a "Cup" or "Chair" is detected with high confidence, the system rejects it immediately.
2. **Classification (MobileNetV2):** The "Brain" layer. The cropped image is passed to a Transfer Learning model fine-tuned on 15,000+ fashion images.
3. **The Guardrail:** A "Smart Threshold" system requires 85% confidence for full-image guesses and 70% for cropped detections, ensuring high reliability.

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow / Keras (MobileNetV2)
- **Object Detection:** Ultralytics (YOLOv8)
- **Computer Vision:** OpenCV
- **Web Framework:** Flask

## 👔 Supported Categories
The model is trained to recognize 10 distinct apparel and accessory classes:
- **Clothing:** Tshirts, Shirts, Tops, Kurtas, Dresses
- **Footwear:** Casual Shoes, Heels
- **Accessories:** Handbags, Watches, Sunglasses
