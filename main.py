import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import cvzone
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, SecondLocator

# MongoDB connection details
connection_string = "mongodb+srv://mark_2:mark100@hukul.cxlxfwa.mongodb.net/numberplate"
database_name = "numberplate"
collection_name = "numberplates"

# Initialize PaddleOCR
ocr = PaddleOCR()

# Load the YOLO model
model = YOLO("best_float32.tflite")

# Load class names
with open("coco1.txt", "r") as f:
    class_names = f.read().splitlines()

def manage_numberplate_db(numberplate, timestamp):
    """Insert the number plate and timestamp into MongoDB."""
    try:
        client = MongoClient(connection_string)
        db = client[database_name]
        collection = db[collection_name]

        data = {
            "numberplate": numberplate,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
        collection.insert_one(data)
        print("Data inserted successfully.")
    except Exception as e:
        print(f"Error inserting data into MongoDB: {e}")
    finally:
        client.close()

def perform_ocr(image_array):
    """Perform OCR on the provided image array."""
    if image_array is None:
        return ""
    results = ocr.ocr(image_array, rec=True)
    detected_text = ''.join(result[1][0] for result in results[0] if results)
    return detected_text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('-', ' ').strip()

# Initialize video capture
cap = cv2.VideoCapture('tc.mp4')

# Area for counting vehicles
area = [(5, 180), (3, 249), (984, 237), (950, 168)]
counter = []
timestamps = []
numberplates = []

# Set up the plot
plt.ion()  # Enable interactive mode
plt.figure(figsize=(10, 6))
plt.title('Detected Number Plates Over Time')
plt.xlabel('Time')
plt.ylabel('Number Plates')
plt.grid()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.track(frame, persist=True, imgsz=240)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            if class_names[class_id] == "numberplate":
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0 and track_id not in counter:
                    counter.append(track_id)
                    cropped_img = frame[y1:y2, x1:x2]
                    cropped_img = cv2.resize(cropped_img, (120, 70))
                    detected_text = perform_ocr(cropped_img)

                    if detected_text:
                        timestamp = datetime.now()
                        manage_numberplate_db(detected_text, timestamp)
                        timestamps.append(timestamp)
                        numberplates.append(detected_text)

    # Plot data with optimization
    if timestamps and numberplates:
        plt.clf()  # Clear the current figure
        plt.plot(timestamps, numberplates, marker='o', linestyle='-', color='b')
        plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(SecondLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.draw()
        plt.pause(0.001)  # Pause to allow the plot to update

    mycounter = len(counter)
    cvzone.putTextRect(frame, f'Count: {mycounter}', (50, 60), 1, 1)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()