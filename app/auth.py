import pyrebase

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

auth = firebase.auth()

def sign_up(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        return {"message": "User created", "user": user}, 200
    except Exception as e:
        return {"error": str(e)}, 400

def sign_in(email, password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        return {"message": "User signed in", "token": user['idToken']}, 200
    except Exception as e:
        return {"error": str(e)}, 400

def forgot_password(email):
    try:
        auth.send_password_reset_email(email)
        return {"message": "Password reset email sent"}, 200
    except Exception as e:
        return {"error": str(e)}, 400
