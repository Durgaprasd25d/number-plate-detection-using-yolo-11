import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
from server import manage_numberplate_db
import cvzone
from datetime import datetime

# Initialize PaddleOCR
ocr = PaddleOCR()
cap = cv2.VideoCapture('tc.mp4')  # Path to the video file
model = YOLO("best_float32.tflite")

# Load class names
with open("coco1.txt", "r") as f:
    class_names = f.read().splitlines()

# Function to perform OCR on an image array
def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")

    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results:
        for result in results[0]:
            text = result[1][0]
            detected_text.append(text)

    # Join all detected texts into a single string
    return ''.join(detected_text)

# Initialize video capture and YOLO model
counter = []
area = [(5, 180), (3, 249), (984, 237), (950, 168)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.track(frame, persist=True, imgsz=240)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            if class_names[class_id] == "numberplate":  # Ensure it's a number plate
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 0 and track_id not in counter:
                    counter.append(track_id)  # Only add if it's a new track ID

                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (120, 70))
                    
                    text = perform_ocr(crop)
                    text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('-', ' ').strip()
                    
                    if text:  # Only manage if text is not empty
                        timestamp = datetime.now()
                        manage_numberplate_db(text, timestamp)

    mycounter = len(counter)               
    cvzone.putTextRect(frame, f'{mycounter}', (50, 60), 1, 1)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Close video capture
cap.release()
cv2.destroyAllWindows()
