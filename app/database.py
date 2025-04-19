from flask_pymongo import PyMongo
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os


# Initialize PyMongo
mongo = PyMongo()

def init_db(app):
    """Initialize MongoDB with Flask app"""
    app.config["MONGO_URI"] = "mongodb+srv://ninitheninja10:NsqD15SPLvr8cXl1@signlingocluster.ssmabvu.mongodb.net/signlingo?retryWrites=true&w=majority&appName=SignLingoCluster"
    mongo.init_app(app)
    
    try:
        # Test the connectio
        print("Successfully connected to MongoDB!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise e 