import cv2
from ultralytics import YOLOv10
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime, timedelta
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create a Video Capture Object
cap = cv2.VideoCapture("./Resources/videoplayback.mp4")

# Initialize the YOLOv10 Model
model = YOLOv10("./weights/best.pt")

# Initialize the frame count
count = 0

# Class Names
className = ["License"]

# Initialize the Paddle OCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)

# Indian License Plate Pattern
indian_license_plate_pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}$')

# Set up SQLite database connection for registered vehicles database
def create_registered_db():
    conn = sqlite3.connect('registered_vehicles.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS RegisteredVehicles(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_plate TEXT UNIQUE
        )
    ''')
    conn.commit()
    conn.close()

# Set up SQLite database connection for vehicle logs (registered/unregistered)
def create_vehicle_logs_db():
    conn = sqlite3.connect('vehicle_logs.db')
    cursor = conn.cursor()

    # Table for registered vehicle logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS RegisteredVehicleLogs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_plate TEXT,
            detection_time TEXT
        )
    ''')

    # Table for unregistered vehicle logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS UnregisteredVehicleLogs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_plate TEXT,
            detection_time TEXT
        )
    ''')

    conn.commit()
    conn.close()

# Check if a vehicle is registered
def is_registered(plate):
    conn = sqlite3.connect('registered_vehicles.db')
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM RegisteredVehicles WHERE license_plate = ?', (plate,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# Save registered vehicles with detection time in vehicle logs database
def save_registered_vehicle_log(plate, detection_time):
    conn = sqlite3.connect('vehicle_logs.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO RegisteredVehicleLogs(license_plate, detection_time)
        VALUES (?, ?)
    ''', (plate, detection_time))

    conn.commit()
    conn.close()

# Save unregistered vehicles with detection time in vehicle logs database
def save_unregistered_vehicle_log(plate, detection_time):
    conn = sqlite3.connect('vehicle_logs.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO UnregisteredVehicleLogs(license_plate, detection_time)
        VALUES (?, ?)
    ''', (plate, detection_time))

    conn.commit()
    conn.close()

# Extract license plate using PaddleOCR
def paddle_ocr(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    result = ocr.ocr(frame, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]

    # Clean and format the extracted text
    pattern = re.compile(r'[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "").replace("O", "0").replace("I", "1")
    
    # Check if the extracted text matches Indian license plate format
    if indian_license_plate_pattern.match(text):
        return str(text)
    
    return ""  # Return an empty string if not valid

# Main logic for the video processing and detection
create_registered_db()  # Ensure the registered vehicle database is created
create_vehicle_logs_db()  # Ensure the vehicle logs database is created

# Dictionary to track the last detection time of each plate
last_detected = {}

while True:
    ret, frame = cap.read()
    if ret:
        currentTime = datetime.now()
        count += 1
        print(f"Frame Number: {count}")
        results = model.predict(frame, conf=0.45)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                classNameInt = int(box.cls[0])
                clsName = className[classNameInt]
                conf = math.ceil(box.conf[0] * 100) / 100
                
                # Run OCR to extract the license plate text
                label = paddle_ocr(frame, x1, y1, x2, y2)
                if label:
                    # Always show the license plate on the video feed
                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    
                    # Determine the color for the bounding box
                    if is_registered(label):
                        color = (0, 255, 0)  # Green for registered
                        # Check if the plate was detected within the last minute for database entry
                        if label not in last_detected or currentTime - last_detected[label] >= timedelta(minutes=1):
                            last_detected[label] = currentTime
                            save_registered_vehicle_log(label, currentTime.isoformat())  # Save to registered logs
                    else:
                        color = (0, 0, 255)  # Red for unregistered
                        # Check if the plate was detected within the last minute for database entry
                        if label not in last_detected or currentTime - last_detected[label] >= timedelta(minutes=1):
                            last_detected[label] = currentTime
                            save_unregistered_vehicle_log(label, currentTime.isoformat())  # Save to unregistered logs
                    
                    cv2.rectangle(frame, (x1, y1), c2, color, -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
