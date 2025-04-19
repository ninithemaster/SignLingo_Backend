from flask import Flask, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename
import PIL.Image
from google import genai
from google.genai import types
from app.database import init_db, mongo
from app.auth import sign_up, sign_in, forgot_password


# Load YOLO model
app = Flask(__name__) 
# Initialize MongoDB
init_db(app)

obj_model = YOLO("yolov10n.pt")
MODEL_PATH = 'sign_language_model.keras'  # or .h5 if you're still using that
IMAGE_SIZE = (128, 128)

# Load model once when Flask starts
model = load_model(MODEL_PATH)

# Class names (same order as your training directory)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']


@app.route('/')
def home():
    return jsonify({"message": "Welcome to my Flask Object Detection API!"})

@app.route('/obj-detection', methods=["POST"])
def obj_detection():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    
    # Convert image file to numpy array
    image_np = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Ensure image is valid
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Run YOLO object detection
    results = obj_model.predict(image)
    detected_objects = [obj_model.names[int(box.cls)] for box in results[0].boxes]
    unique_detected_objects = list(set(detected_objects))
    return jsonify({"detections": unique_detected_objects})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)

    # Save uploaded file
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    # Read & preprocess
    img = cv2.imread(filepath)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    predicted_class = class_names[np.argmax(preds)]

    # Clean up
    os.remove(filepath)

    return jsonify({
        'prediction': predicted_class,
        'confidence': float(np.max(preds))
    })


@app.route("/gemini-detection", methods=["POST"])
def gemini_detection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    mime_type = image_file.mimetype or mimetypes.guess_type(image_file.filename)[0]

    if not mime_type:
        return jsonify({'error': 'Could not determine MIME type'}), 400

    prompt = "Identify the object shown in the image. Just return the object's name. Do not explain."

    # Prepare image as Gemini Part
    part = {
        "inline_data": {
            "mime_type": mime_type,
            "data": image_bytes
        }
    }

    # Gemini call
    client = genai.Client(api_key="AIzaSyCoMjeQ3GL6Y69HG1xwJB9qM_FTlXpsbIY")
    response = client.models.generate_content(
    model="gemini-2.5-pro-exp-03-25",
    contents=[part, prompt],
)
    detected_objects = response.text
    unique_detected_objects = list(detected_objects.split(" "))
    return jsonify({'prediction': unique_detected_objects})


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json(force=True)  # <-- Force JSON parsing

    print("Received signup data:", data)  # Debug print

    if not all(key in data for key in ['email', 'password', 'name']):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        response, status = sign_up(data['email'], data['password'], data['name'])
        return jsonify(response), status
    except Exception as e:
        print("Error in sign_up:", e)
        return jsonify({"error": "Server error occurred"}), 500


@app.route('/signin', methods=['POST'])
def signin():
    data = request.json
    response, status = sign_in(data['email'], data['password'])
    return jsonify(response), status

@app.route('/forgot-password', methods=['POST'])
def forgotpassword():
    data = request.json
    response, status = forgot_password(data['email'])
    return jsonify(response), status







