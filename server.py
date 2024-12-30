from pymongo import MongoClient
from datetime import datetime

def manage_numberplate_db(numberplate):
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

        # Step 2: Access the database (it will be created if it doesn't exist)
        db = client[database_name]

        # Step 3: Access the collection (it will be created if it doesn't exist)
        collection = db[collection_name]

        # Step 4: Insert data into the collection
        current_date = datetime.now().date()  # Get current date
        current_time = datetime.now().time()  # Get current time
        data = {
            "numberplate": numberplate,
            "entry_date": str(current_date),
            "entry_time": str(current_time)
        }

        print("Inserting data into the collection...")
        collection.insert_one(data)
        print("Data inserted successfully")

        # Step 5: Retrieve and display data from the collection
        print("Fetching data from the collection...")
        result = collection.find()
        for document in result:
            print(document)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the connection
        client.close()
        print("MongoDB connection is closed")

# Example usage
# manage_numberplate_db("ABC-1234")
