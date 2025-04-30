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
from datetime import datetime, timedelta
import pytz

# Time tracker storage (for simplicity, we will use an in-memory dictionary)
# You can replace this with a database if needed.
time_tracker_data = {}

# Load YOLO model
app = Flask(__name__) 
# Initialize MongoDB
init_db(app)

obj_model = YOLO("model/yolov10n.pt")
MODEL_PATH = 'model/sign_language_model.keras'  # or .h5 if you're still using that
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

@app.route("/detect_object", methods=["POST"])
def detect_objects():
    """Try YOLO detection first, fallback to Gemini if YOLO detects nothing."""
    try:
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
        unique_yolo = list(set(detected_objects))

        if unique_yolo:
            return (
                jsonify(
                    {
                        "status": "success",
                        "message": "Identified by YOLO",
                        "name": unique_yolo,
                    }
                ),
                200,
            )

        image_file.stream.seek(0)
        pil_image = PIL.Image.open(image_file)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing in the environment.")

        client = genai.Client(api_key=api_key)
        prompt = "Just state the object name, do not form any sentence."
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[pil_image, prompt],
        )

        gemini_detected = response.text.strip().split(" ")
        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Identified by Gemini",
                    "name": gemini_detected,
                }
            ),
            200,
        )

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/gemini-sign-predict", methods=["POST"])
def gemini_sign_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    mime_type = image_file.mimetype or mimetypes.guess_type(image_file.filename)[0]

    if not mime_type:
        return jsonify({'error': 'Could not determine MIME type'}), 400

    prompt = (
        "Identify the American Sign Language (ASL) letter, number, or word shown in the image. "
        "Just return the recognized letter, number, or word. Do not provide any explanation."
        "Prioritize word, then letters or numbers"
    )

    part = {
        "inline_data": {
            "mime_type": mime_type,
            "data": image_bytes
        }
    }

    client = genai.Client(api_key="AIzaSyCoMjeQ3GL6Y69HG1xwJB9qM_FTlXpsbIY")
    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=[part, prompt],
    )

    detected_sign = response.text.strip()
    return jsonify({'prediction': detected_sign}),200

@app.route('/start-task', methods=['POST'])
def start_task():
    task_data = request.get_json(force=True)
    task_name = task_data.get("task_name")
    user_id = task_data.get("user_id")
    
    if not task_name or not user_id:
        return jsonify({"error": "Task name and user_id are required"}), 400
    
    # Get the user's last task data
    user_task_record = mongo.db.tasks.find_one({"user_id": user_id, "end_time": None})
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Initialize streak and level variables
    streak = 1
    level = 1
    
    if user_task_record:
        # Check if the previous task was completed today
        last_task_date = datetime.strptime(user_task_record["start_time"], "%Y-%m-%dT%H:%M:%S.%fZ").date()
        if last_task_date == datetime.now().date() - timedelta(days=1):
            # The task is consecutive, increment streak
            streak = user_task_record.get("streak", 1) + 1
        else:
            # The streak was broken, reset to 1
            streak = 1

    # Calculate user level based on streak
    level = (streak // 3) + 1  # Example: 3-day streak = level 2, 6-day streak = level 3
    
    # Save the task with the streak and level info
    start_time = datetime.now(pytz.utc).isoformat()  # Current time in ISO format
    task_record = {
        "task_name": task_name,
        "start_time": start_time,
        "user_id": user_id,
        "streak": streak,
        "level": level,
        "end_time": None,
        "duration": None
    }
    
    try:
        mongo.db.tasks.insert_one(task_record)
        return jsonify({
            "message": f"Started tracking for task: {task_name}",
            "streak": streak,
            "level": level,
            "start_time": start_time
        }), 200
    except Exception as e:
        print(f"Error inserting task data into database: {e}")
        return jsonify({"error": "Database error occurred"}), 500


@app.route('/end-task', methods=['POST'])
def end_task():
    task_data = request.get_json(force=True)
    task_name = task_data.get("task_name")
    user_id = task_data.get("user_id")
    
    if not task_name or not user_id:
        return jsonify({"error": "Task name and user_id are required"}), 400
    
    task_record = mongo.db.tasks.find_one({"task_name": task_name, "user_id": user_id, "end_time": None})
    
    if not task_record:
        return jsonify({"error": "Task not found or already ended"}), 400
    
    # Calculate duration
    start_time = datetime.fromisoformat(task_record["start_time"])  # Parse ISO string
    end_time = datetime.now(pytz.utc).isoformat()  # Current time in ISO format
    end_time_dt = datetime.fromisoformat(end_time)  # Parse ISO string for end time
    duration = (end_time_dt - start_time).total_seconds() / 60  # duration in minutes
    
    # Update task record with end time and duration
    mongo.db.tasks.update_one(
        {"_id": task_record["_id"]},
        {"$set": {"end_time": end_time, "duration": duration}}
    )
    
    # Calculate streak and level
    streak = task_record["streak"]
    level = task_record["level"]
    
    # Calculate new level based on completed tasks or streak
    if streak % 3 == 0:
        level += 1  # Increase level for every 3-day streak
    
    # Update level in database
    mongo.db.tasks.update_one(
        {"_id": task_record["_id"]},
        {"$set": {"streak": streak, "level": level}}
    )
    
    return jsonify({
        "message": f"Ended tracking for task: {task_name}",
        "end_time": end_time,
        "duration_minutes": duration,
        "streak": streak,
        "level": level
    }), 200


@app.route('/get-user-status', methods=['GET'])
def get_user_status():
    user_id = request.args.get("user_id")
    
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    # Retrieve the user's latest task
    user_task = mongo.db.tasks.find_one({"user_id": user_id, "end_time": None})
    
    if not user_task:
        return jsonify({"error": "No active task found for the user"}), 400
    
    return jsonify({
        "user_id": user_id,
        "streak": user_task.get("streak", 1),
        "level": user_task.get("level", 1),
        "current_task": user_task["task_name"],
        "start_time": user_task["start_time"]
    }), 200
    

# Load questions and answers from JSON file
def load_questions():
    with open("model/questions.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.route('/quiz', methods=['GET'])
def get_quiz():
    questions = load_questions()

    # Remove answers before sending to frontend
    quiz_questions = [
        {
            "id": q["id"],
            "question": q["question"],
            "options": q["options"]
        } for q in questions
    ]
    return jsonify({"quiz": quiz_questions}), 200

@app.route('/submit-quiz', methods=['POST'])
def submit_quiz():
    submitted_data = request.get_json(force=True)

    if "answers" not in submitted_data:
        return jsonify({"error": "Missing answers in request"}), 400

    user_answers = submitted_data["answers"]  # {1: "option text", 2: "option text", ...}

    questions = load_questions()
    answer_key = {q["id"]: q["answer"] for q in questions}

    correct_count = 0
    detailed_results = []

    for qid, user_answer in user_answers.items():
        correct_answer = answer_key.get(int(qid))
        is_correct = user_answer == correct_answer
        if is_correct:
            correct_count += 1
        detailed_results.append({
            "question_id": int(qid),
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })

    return jsonify({
        "score": correct_count,
        "total": len(answer_key),
        "results": detailed_results
    }), 200


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







