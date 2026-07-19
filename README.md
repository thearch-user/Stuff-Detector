# Stuff Detector

Stuff Detector is a custom image classification system built with **TensorFlow/Keras**, **OpenCV**, and **MobileNetV3 transfer learning**.

The project uses a pretrained MobileNetV3 model as a feature extractor. The original ImageNet classification head is removed and replaced with a custom classification head trained to recognize user-defined object classes.

The project includes:

- A machine learning pipeline
- Image and camera processing
- A Flask API backend
- A browser-based frontend

---

# Features

## Machine Learning

- MobileNetV3 transfer learning
- Custom classification head
- User-defined object classes
- TensorFlow/Keras model training
- Image preprocessing pipeline
- Real-time inference

## Application

- Webcam classification
- Flask REST API
- Browser frontend
- JSON prediction responses
- Expandable dataset system

---

# How It Works

Stuff Detector does not train the entire neural network from scratch.

Instead, it uses transfer learning:

```
Input Image
     |
     v
MobileNetV3 Backbone
     |
     v
Remove Original ImageNet Classification Head
     |
     v
Custom Classification Layers
     |
     v
Object Prediction
```

The pretrained MobileNetV3 layers extract useful visual features while the custom classifier learns to recognize objects from the project's dataset.

---

# Architecture

```
                    Browser

                       |
                       |
                       v

              frontend/index.html

                       |
                       |
              HTTP POST /detect

                       |
                       v

               backend/app.py

                       |
                       |
              Detection Pipeline

                       |
                       v

              TensorFlow Model

                       |
                       v

               Prediction Result
```

---

# Project Structure

```
Stuff-Detector/

├── requirements.txt
│
├── main.py
├── detector.py
├── camera.py
├── image_loader.py
├── image_shower.py
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── index.html
│
├── data/
│   └── images/
│
└── models/
    └── model.keras
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/thearch-user/Stuff-Detector.git

cd Stuff-Detector
```

---

## Install Dependencies

Install packages from the root requirements file:

```bash
pip install -r requirements.txt
```

---

# Dataset Structure

Training images are stored inside:

```
data/images/
```

Each folder represents a class.

Example:

```
data/images/

├── keyboard/
│   ├── image1.jpg
│   ├── image2.jpg
│
├── bottle/
│   ├── image1.jpg
│   └── image2.jpg
```

The folder name becomes the class label.

---

# Model Training

The training process:

1. Load images from the dataset
2. Resize images
3. Apply MobileNetV3 preprocessing
4. Use pretrained MobileNetV3 feature extraction
5. Train a custom classification head
6. Save the trained model

The final model is used during inference.

---

# Running The Application

## Start Backend

From the project root:

```bash
python backend/app.py
```

The Flask server will start:

```
http://localhost:5000
```

---

## Start Frontend

Open:

```
frontend/index.html
```

in a browser.

The frontend will:

1. Request camera access
2. Capture frames
3. Send images to Flask
4. Display predictions

---

# Backend API

The Flask backend acts as the bridge between the frontend and the machine learning model.

It handles:

- Receiving images
- Processing requests
- Running inference
- Returning prediction results

---

# API Endpoints

## Health Check

### GET /

Example:

```
GET http://localhost:5000/
```

Response:

```json
{
    "status": "running"
}
```

---

## Detect Object

### POST /detect

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
    "confidence": 0.95
}
```

---

# Adding New Objects

To add a new object:

1. Add training images:

```
data/images/new_object/
```

Example:

```
data/images/mouse/

├── image1.jpg
├── image2.jpg
└── image3.jpg
```

2. Retrain the model.

3. Replace the existing model file.

4. Restart the backend.

---

# Technologies Used

## Machine Learning

| Technology | Purpose |
|---|---|
| TensorFlow | Deep learning framework |
| Keras | Model building |
| MobileNetV3 | Feature extractor |
| OpenCV | Image processing |
| NumPy | Data processing |

## Backend

| Technology | Purpose |
|---|---|
| Flask | API server |
| Flask-CORS | Frontend communication |

## Frontend

| Technology | Purpose |
|---|---|
| HTML | User interface |
| JavaScript | Webcam handling |
| Browser Media API | Camera access |

---

# Limitations

- Only performs classification, not object detection
- Requires retraining for new classes
- Accuracy depends on dataset quality
- Lighting conditions can affect predictions
- No bounding box detection

---

# Future Improvements

- Real-time video streaming
- Better frontend interface
- Automatic dataset collection
- Model optimization
- Mobile deployment
- Object detection with bounding boxes
- Cloud deployment

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

Push changes:

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