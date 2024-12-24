
from pymongo.mongo_client import MongoClient
from diabetes.logging.logger import logging 

uri = "mongodb+srv://sivakumarreddym22:siva123@cluster0.5d1wc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

logging.info("mongo tested successfully")