# Stuff Detector

A custom object recognition system built with **TensorFlow/Keras**, **MobileNetV3 transfer learning**, **OpenCV**, and **Flask**.

Stuff Detector uses a pretrained **MobileNetV3-Large** model, removes the original ImageNet classification head, and replaces it with a custom classification head trained to recognize user-defined objects.

The project allows you to train your own object classifier using your own dataset while leveraging powerful pretrained computer vision features.

---

# Features

## Machine Learning

- Transfer learning using MobileNetV3-Large
- Removed ImageNet classification head
- Custom classification head trained from scratch
- User-defined object classes
- TensorFlow/Keras model training
- Real-time inference

## Application

- Webcam image classification
- Flask REST API backend
- Browser-based frontend
- JSON prediction responses
- Easy dataset expansion

---

# How It Works

The model uses transfer learning.

Instead of training an entire neural network from zero, Stuff Detector uses MobileNetV3 as a feature extractor.

```
Input Image
     |
     v
MobileNetV3-Large Backbone
     |
     v
Remove Original ImageNet Head
     |
     v
Custom Classification Head
     |
     v
Object Prediction
```

The pretrained layers understand general visual features such as:

- edges
- textures
- shapes
- patterns

The custom classification layers learn how to identify your own objects.

---

# Architecture

```
                 MobileNetV3-Large
                 (Pretrained ImageNet)

                         |
                         |
              Original classifier removed

                         |
                         v

              Custom Dense Classifier

                         |
                         v

              User-defined object classes
```

---

# Project Structure

```
Stuff-Detector/

│
├── backend/
│   │
│   ├── app.py
│   │   Flask API server
│   │
│   ├── requirements.txt
│   │   Python dependencies
│   │
│   └── models/
│       └── stuff_detector.keras
│           Trained TensorFlow model
│
├── frontend/
│   └── index.html
│       Browser webcam interface
│
├── data/
│   └── images/
│       Training dataset
│
├── detector.py
│   Model inference logic
│
├── camera.py
│   Camera handling
│
├── image_loader.py
│   Image loading utilities
│
├── image_shower.py
│   Image visualization
│
├── main.py
│   Main application
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/thearch-user/Stuff-Detector.git

cd Stuff-Detector
```

---

## Create Virtual Environment

Linux/macOS:

```bash
python3 -m venv venv

source venv/bin/activate
```

Windows:

```bash
python -m venv venv

venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

# Dataset Format

Training data should be organised by class.

Example:

```
data/images/

├── bottle/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
│
├── keyboard/
│   ├── image1.jpg
│   └── image2.jpg
│
└── phone/
    ├── image1.jpg
    └── image2.jpg
```

Each folder represents a class.

Example:

```
data/images/keyboard/
```

creates the class:

```
keyboard
```

---

# Training

The training pipeline:

1. Load images from dataset
2. Resize images
3. Apply MobileNetV3 preprocessing
4. Use pretrained MobileNetV3 feature extraction
5. Train custom classification layers
6. Save trained model

The final model is saved as:

```
backend/models/stuff_detector.keras
```

---

# Running The Application

## Start Backend

Navigate to backend:

```bash
cd backend
```

Run Flask:

```bash
python app.py
```

The API starts at:

```
http://localhost:5000
```

---

## Start Frontend

Open:

```
frontend/index.html
```

in your browser.

Allow camera permissions.

The frontend will:

1. Access webcam
2. Capture frames
3. Send images to Flask
4. Display prediction results

---

# API Documentation

## Health Check

### GET /

Request:

```
GET localhost:5000/
```

Response:

```json
{
    "status": "online",
    "model": "Stuff Detector"
}
```

---

# Prediction Endpoint

## POST /detect

Classifies an image.

Request:

```json
{
    "image": "base64_encoded_image"
}
```

Response:

```json
{
    "label": "keyboard",
    "confidence": 0.9432
}
```

---

# Adding New Objects

To add a new object:

## 1. Add images

Example:

```
data/images/mouse/
```

Add:

```
mouse1.jpg
mouse2.jpg
mouse3.jpg
```

---

## 2. Retrain Model

The classifier must be retrained because the output classes changed.

---

## 3. Replace Model

Place the new model:

```
backend/models/stuff_detector.keras
```

Restart Flask.

---

# Requirements

- Python 3.10+
- TensorFlow
- Keras
- OpenCV
- Flask
- NumPy
- Pillow

Recommended:

- NVIDIA GPU
- CUDA support

---

# Technologies

## Machine Learning

| Technology | Purpose |
|-|-|
| TensorFlow | Deep learning framework |
| Keras | Neural network API |
| MobileNetV3 | Feature extractor |
| OpenCV | Image processing |
| NumPy | Data processing |

## Backend

| Technology | Purpose |
|-|-|
| Flask | REST API |
| Flask-CORS | Frontend communication |

## Frontend

| Technology | Purpose |
|-|-|
| HTML | Interface |
| JavaScript | Webcam handling |
| Browser Media API | Camera access |

---

# Limitations

Current limitations:

- Only classifies the main object
- Requires retraining for new categories
- No object bounding boxes
- Accuracy depends on dataset quality
- Lighting affects predictions

---

# Future Improvements

- Real-time WebSocket streaming
- Object detection with bounding boxes
- Automatic dataset collection
- Model quantization
- Mobile deployment
- Better confidence filtering
- Cloud hosting

---

# Contributing

Contributions are welcome.

Create a branch:

```bash
git checkout -b feature-name
```

Commit changes:

```bash
git commit -m "Add feature"
```

Push:

```bash
git push origin feature-name
```

Open a pull request.

---

# License

MIT License

---

# Author

Created by **thearch-user**