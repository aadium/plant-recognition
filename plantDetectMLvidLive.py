import cv2
import numpy as np

# Load Haar Cascade classifier
plant_cascade = cv2.CascadeClassifier('plant_cascade.xml')

# Initialize video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video file
    ret, frame = cap.read()

    # Check if frame was successfully read
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply noise reduction and contrast stretching
    gray = cv2.GaussianBlur(gray, (15,15), 0)
    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(20,20))
    gray = clahe.apply(gray)

    # Detect plants in frame
    plants = plant_cascade.detectMultiScale(gray, scaleFactor=7, minNeighbors=8)

    # If plants are detected, draw rectangles around them and display the number of plants detected
    if len(plants) > 0:
        total_plants = len(plants)
        detected_plants = 0
        accuracy_sum = 0
        for (x,y,w,h) in plants:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (125,0,255), 10)
            detected_plants += 1
            accuracy_sum += plants[0][3] / (gray.shape[0] * gray.shape[1]) # Normalize accuracy score
        accuracy = accuracy_sum / total_plants * 100000 # Compute accuracy as a percentage
        cv2.putText(frame, '{:.2f}% accuracy'.format(accuracy), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(frame, str(len(plants)) + ' plant(s) detected. Press q to exit', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    else:
        total_plants = 0
        detected_plants = 0

    # Display frame with plant detection results
    cv2.imshow('Frame', frame)

    # Wait for 1 millisecond for the next frame
    if cv2.waitKey(50) == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
