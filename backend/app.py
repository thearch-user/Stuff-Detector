import os
import json
import base64

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


MODEL_PATH = "models/stuff_detector.keras"
CLASSES_PATH = "../classes.json"


# Load model
print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded")


# Load class names
with open(CLASSES_PATH, "r") as file:
    classes = json.load(file)


IMG_SIZE = (224, 224)


def preprocess(image):

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        "float32"
    )

    image = tf.keras.applications.mobilenet_v3.preprocess_input(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image



@app.route("/")
def home():

    return {
        "status": "online",
        "model": "Stuff Detector"
    }



@app.route("/detect", methods=["POST"])
def detect():

    data = request.json


    if "image" not in data:
        return jsonify({
            "error": "Missing image"
        }),400



    encoded = data["image"].split(",")[1]


    img_bytes = base64.b64decode(
        encoded
    )


    np_img = np.frombuffer(
        img_bytes,
        np.uint8
    )


    frame = cv2.imdecode(
        np_img,
        cv2.IMREAD_COLOR
    )


    processed = preprocess(
        frame
    )


    prediction = model.predict(
        processed
    )


    index = int(
        np.argmax(prediction)
    )


    confidence = float(
        np.max(prediction)
    )


    label = classes[index]


    return jsonify({

        "label": label,

        "confidence":
        round(confidence,4)

    })



if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )