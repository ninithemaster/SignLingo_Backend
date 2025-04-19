import pyrebase
from app.database import mongo

firebaseConfig = {
  "apiKey": "AIzaSyAETwrFzzbwIEJ8SeHTUp9TgOHr5YAE6_E",
  "authDomain": "signlangauth-147a5.firebaseapp.com",
  "projectId": "signlangauth-147a5",
  "storageBucket": "signlangauth-147a5.firebasestorage.app",
  "databaseURL":"https://signlangauth-147a5-default-rtdb.asia-southeast1.firebasedatabase.app",
  "messagingSenderId": "57365397215",
  "appId": "1:57365397215:web:e1451d537ddee18022336a",
  "measurementId": "G-5TRGGGCLV1"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()


def sign_up(email, password, name):
    try:
        # Create user in Firebase
        user = auth.create_user_with_email_and_password(email, password)
        
        # Store user data in MongoDB
        user_data = {
            "email": email,
            "name": name,
            "firebase_uid": user['localId']
        }
        mongo.db.users.insert_one(user_data)
        
        return {"message": "User created", "user": user}, 200
    except Exception as e:
        return {"error": str(e)}, 400
    if "EMAIL_EXISTS" in error_message:
            return {"error": "Email already exists. Please sign in."}, 409
    elif "INVALID_EMAIL" in error_message:
            return {"error": "Invalid email format."}, 400
    else:
            return {"error": error_message}, 400

def sign_in(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        # Get user data from MongoDB
        user_data = mongo.db.users.find_one({"email": email})
        if user_data:
            user_data['_id'] = str(user_data['_id'])  # Convert ObjectId to string
            return {
                "message": "User signed in", 
                "token": user['idToken'], 
                "user": {
                    "email": user_data['email'],
                    "name": user_data['name'],
                    "id": user_data['_id']
                }
            }, 200
        return {"error": "User data not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 400

def forgot_password(email):
    try:
        auth.send_password_reset_email(email)
        return {"message": "Password reset email sent"}, 200
    except Exception as e:
        return {"error": str(e)}, 400
