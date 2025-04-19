from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Replace with your MongoDB URI
MONGO_URI = "mongodb+srv://ninitheninja10:NsqD15SPLvr8cXl1@signlingocluster.ssmabvu.mongodb.net/?retryWrites=true&w=majority&appName=SignLingoCluster"  # or your MongoDB Atlas URI

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)  # 5 second timeout
    # Attempt to get server info
    client.server_info()
    print("✅ Successfully connected to MongoDB!")
except ConnectionFailure as e:
    print("❌ Could not connect to MongoDB:", e)
