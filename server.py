from pymongo import MongoClient
from datetime import datetime

def manage_numberplate_db(numberplate, timestamp):
    """Connect to MongoDB, create database, create collection, insert number plate, and fetch data."""

    # MongoDB connection string
    connection_string = "mongodb+srv://mark_2:mark100@hukul.cxlxfwa.mongodb.net/numberplate"
    database_name = "numberplate"
    collection_name = "numberplates"

    try:
        # Step 1: Connect to MongoDB Server
        print("Attempting to connect to MongoDB Server...")
        client = MongoClient(connection_string)
        print("Connection to MongoDB Server successful")

        # Step 2: Access the database
        db = client[database_name]

        # Step 3: Access the collection
        collection = db[collection_name]

        # Step 4: Insert data into the collection
        data = {
            "numberplate": numberplate,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")  # Format timestamp
        }

        print("Inserting data into the collection...")
        collection.insert_one(data)
        print("Data inserted successfully")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the connection
        client.close()
        print("MongoDB connection is closed")