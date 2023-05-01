import cv2
import numpy as np
import matplotlib.pyplot as plt

fileName = input("Please enter the name of the file with the extension: ")

# Load image
img = cv2.imread('images/' + fileName)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply noise reduction and contrast stretching
gray = cv2.GaussianBlur(gray, (13,13), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# Load Haar Cascade classifier
plant_cascade = cv2.CascadeClassifier('plant_cascade.xml')

# Detect plants in image
plants = plant_cascade.detectMultiScale(gray, scaleFactor=6, minNeighbors=5)

# If plants are detected, draw rectangles around them and display the number of plants detected
if len(plants) > 0:
    for (x,y,w,h) in plants:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 5)

    # Display image with detected faces and number of faces detected
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(str(len(plants)) + ' plant(s) detected')
    plt.show()
    
# If no faces are detected, display a message
else:
    print("No plants detected in the image.")
