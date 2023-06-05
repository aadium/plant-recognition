import cv2
import numpy as np

# Load Haar Cascade classifier
plant_cascade = cv2.CascadeClassifier('plant_cascade.xml')

# Initialize video capture object
# liveFeed = cv2.VideoCapture("v.mp4")
liveFeed = cv2.VideoCapture(0)

# Define the size of the spray area
spray_radius = 50

while True:
    # Read frame from video file
    read, frame = liveFeed.read()

    # Check if frame was successfully read
    if not read:
        break

    # Convert image to HSV color space (to separate the hue, saturation, and intensity)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create color mask (Isolating the green pixels)
    lowerGreen = np.array([30, 50, 50])
    upperGreen = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lowerGreen, upperGreen)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Ignore contours that are too small
        if area < 200:
            continue

        # Draw the contour
        cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)

    # Detect plants in frame
    plants = plant_cascade.detectMultiScale(frame, scaleFactor=5, minNeighbors=6)

    # If plants are detected, draw rectangles around them and display the number of plants detected
    if len(plants) > 0:
        for (x,y,w,h) in plants:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 10)
        cv2.putText(frame, str(len(plants)) + ' plant(s) detected. Press q to exit', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # Nozzle will spray only if contours and plants are detected
    if (len(contours)>0 and len(plants)>0):
        for contour in contours:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw a circle around the centroid
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Turn on the nozzle if the centroid is within the spray area
                if cx > spray_radius and cx < frame.shape[1] - spray_radius and \
                cy > spray_radius and cy < frame.shape[0] - spray_radius:
                    print("Nozzle on")# code to turn the nozzle on goes here
                else:
                        print("Nozzle off")
                        # code to turn the nozzle off goes here

    # Display frame with plant detection results
    cv2.imshow('Frame', frame)

    # Wait for 1 millisecond for the next frame
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture object and close all windows
liveFeed.release()
cv2.destroyAllWindows()