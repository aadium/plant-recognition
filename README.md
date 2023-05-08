# Plant recognition
This code is an example of using OpenCV (Open Source Computer Vision Library) to detect and track plants in real-time video frames using a Haar Cascade classifier. The code utilizes the Haar Cascade classifier trained specifically for plant detection.

Importing Libraries:
<li>The code starts by importing the necessary libraries, namely cv2 for OpenCV and numpy for numerical operations.
Loading the Haar Cascade Classifier:

<li>The Haar Cascade classifier is loaded using cv2.CascadeClassifier('plant_cascade.xml'). The XML file contains the trained classifier parameters for detecting plants.

Initializing Video Capture:
<li>The code initializes the video capture object using cv2.VideoCapture(0), which captures video from the default camera (0).

Main Loop:
<li>The main loop of the code starts with a while True loop that continuously reads frames from the video capture object until interrupted by pressing 'q' key.
<li>Inside the loop, a frame is read from the video capture object using cap.read().
The code checks if the frame was successfully read using the variable ret. If ret is False, it means there are no more frames, and the loop breaks.

Preprocessing the Frame:
<li>The frame is converted to grayscale using cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY). Grayscale images are commonly used in computer vision tasks and reduce the computational complexity.
<li>Noise reduction and contrast stretching are applied to the grayscale frame. cv2.GaussianBlur is used to apply Gaussian blurring with a kernel size of (15,15). This helps to reduce noise and smooth out the image.
<li>The next step is to enhance the contrast using the Contrast Limited Adaptive Histogram Equalization (CLAHE) technique. cv2.createCLAHE is used to create a CLAHE object with a clip limit of 15.0 and a tile grid size of (20,20). Then clahe.apply is used to apply CLAHE to the grayscale frame.

Plant Detection:
<li>The code uses the plant_cascade Haar Cascade classifier to detect plants in the preprocessed grayscale frame. plant_cascade.detectMultiScale is used to detect the plants, and the resulting bounding boxes are stored in the plants variable.

Drawing Bounding Boxes:
<li>If plants are detected (i.e., len(plants) > 0), the code enters a loop to draw rectangles around each detected plant. It also keeps track of the total number of plants detected and the number of plants being processed.
<li>The cv2.rectangle function is used to draw rectangles around the plants using the (x, y, w, h) coordinates obtained from plants. The rectangles are drawn with a color of (125, 0, 255) and a thickness of 10 pixels.
<li>Additionally, the code calculates an accuracy score for each detected plant by normalizing the plant's height (plants[0][3]) with respect to the size of the grayscale frame. The accuracy scores are summed up and later averaged.
<li>Finally, the code computes the overall accuracy as a percentage, displays it along with the number of detected plants on the frame using cv2.putText.

Displaying the Frame:
<li>The frame with the detected plants and related information is displayed using cv2.imshow('Frame', frame).

Exiting the Loop:
<li>The code waits for 50 milliseconds for the next frame using cv2.waitKey(50).
<li>If the 'q' key is pressed during this interval, the program exits the loop.