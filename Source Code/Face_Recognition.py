import cv2
import os
import numpy as np
import sqlite3
from datetime import datetime

# Path to the training data
path = "D:/Face_Recognition/Faces"

# List of directories (each directory contains images of a person)
people = os.listdir(path)

# Dictionary to map each person's name to a unique integer label
label_dict = {}
label_id = 0

# List to store the face images and labels
images = []
labels = []

# Loop through each person's directory
for person in people:
    # Get the path to the person's images
    person_path = os.path.join(path, person)
    
    # Loop through each image
    for image_file in os.listdir(person_path):
        # Load the image file
        image_path = os.path.join(person_path, image_file)
        image = cv2.imread(image_path)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect the face region in the image
        face_cascade = cv2.CascadeClassifier("D:/Face_Recognition/Haar Cascade Classifiers/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        # Loop through each face and add it to the images list
        for (x, y, w, h) in faces:
            face_image = gray[y:y+h, x:x+w]
            images.append(face_image)
            
            # Check if the person's name is already in the label dictionaryq
            if person not in label_dict:
                # Assign a new integer label to the person
                label_dict[person] = label_id
                label_id += 1
                
            # Add the integer label to the labels list
            labels.append(label_dict[person])

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer with the images and labels
recognizer.train(images, np.array(labels))

# Connect to the attendance database
conn = sqlite3.connect("D:\Face_Recognition\Attendance.db")
c = conn.cursor()

# Create the table if it does not exist
c.execute('''CREATE TABLE IF NOT EXISTS attendance
             (name TEXT, date TEXT, time TEXT)''')

# Set the minimum confidence level for face recognition
confidence_threshold = 50

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    print(ret)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    
    # Detect the face regions in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    

    name = ""
    # Loop through each face in the frame
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_image = gray[y:y+h, x:x+w]
        
        # Calculate the LBPH histogram of the face
        label, confidence = recognizer.predict(face_image)
        print(confidence)

        
        # Check if the face matches a trained face in the database
        if confidence > confidence_threshold:
            # Get the name of the recognized person from the label dictionary
            name = [k for k, v in label_dict.items() if v == label][0]
            
        # Get the current date and time
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        # Insert the attendance record into the database
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
        conn.commit()

        # Draw a rectangle around the face in the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the name of the recognized person and the confidence level
        text = f"{name} ({int(confidence)}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('frame', frame)

    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
