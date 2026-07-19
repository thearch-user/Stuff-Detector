README.md
# Stuff Detector

Stuff Detector is an image classification system built using transfer learning with MobileNetV3-Large.

The project uses a pretrained MobileNetV3 backbone, removes the original ImageNet classification head, and replaces it with a custom classifier trained to recognize user-defined objects.

---

# Features

- Transfer learning with MobileNetV3-Large
- Custom object classification
- Webcam based detection
- Flask REST API
- Browser frontend
- Supports adding new object classes through training data

---

# How It Works



Image
|
v
MobileNetV3-Large Backbone
|
v
Removed ImageNet Classification Head
|
v
Custom Classification Layers
|
v
Object Prediction



The model learns project-specific classes while using pretrained visual features from ImageNet.

---

# Project Structure


Stuff-Detector/

├── backend/
│ ├── app.py
│ ├── requirements.txt
│ └── models/
│ └── stuff_detector.keras
│
├── frontend/
│ └── index.html
│
├── data/
│ └── images/
│
├── classes.json
├── detector.py
├── camera.py
└── main.py


---

# Installation


Clone repository:

```bash
git clone https://github.com/thearch-user/Stuff-Detector.git

cd Stuff-Detector

Create environment:

python -m venv venv

Activate:

Linux:

source venv/bin/activate

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r backend/requirements.txt
Running
Start Backend
cd backend

python app.py

Server:

http://localhost:5000
Start Frontend

Open:

frontend/index.html

Allow camera permissions.

The browser will send frames to Flask for classification.

API
POST /detect

Accepts:

{
"image":"base64 encoded image"
}

Returns:

{
"label":"keyboard",
"confidence":0.94
}
Training

Object images are stored inside:

data/images/

Example:

data/images/

├── keyboard/
│   ├── image1.jpg
│   └── image2.jpg

├── bottle/
│   ├── image1.jpg
│   └── image2.jpg


The dataset is used to train the custom classification head.

Technologies
Python
TensorFlow
Keras
MobileNetV3
OpenCV
Flask
HTML/CSS/JavaScript
Future Improvements
Real-time WebSocket streaming
Object bounding boxes
Mobile deployment
Model quantization
Training dashboard