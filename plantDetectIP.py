import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fileName = input("Please enter the name of the file with the extension: ")

# Load image
img = cv2.imread('gPothos/' + fileName)

# Convert image to HSV color space (to separate the hue, saturation, and intensity)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create color mask (Isolating the green pixels)
lower_green = np.array([25, 50, 50])
upper_green = np.array([90, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# Apply morphological operations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and filter out small ones
rect_count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        # Draw a bounding box around the plant
        x, y, w, h = cv2.boundingRect(cnt)
        rect_count = rect_count + 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 255, 100), 20)

# Show image with bounding box around plant
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title(str(rect_count) + ' plant(s) detected')
plt.show()