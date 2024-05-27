import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load YOLO model
model = YOLO('MODELS/best-tune-70.pt')

def RGB(event, x, y, flags, param):
    """
    Callback function to get RGB values when mouse moves over the window.

    Args:
        event (int): Type of mouse event.
        x (int): x-coordinate of the mouse pointer.
        y (int): y-coordinate of the mouse pointer.
        flags (int): Additional flags.
        param: Additional parameters.
    """
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

# Create a window and set mouse callback function
cv2.namedWindow('AREA')
cv2.setMouseCallback('AREA', RGB)

# Open video capture
cap = cv2.VideoCapture(1)

# Read class names from file
with open("smartview_classes.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Define lists to store bounding boxes for different classes
bounding_list = []
bounding_list1 = []
bounding_list2 = []

# Define area of interest polygon
area = [(0, 310), (0, 370), (628, 390), (615, 375)]
# Store counted areas
area_c = set()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (640, 640))

    # Predict using YOLO model
    results = model.predict(frame)
    boxes = results[0].boxes.boxes

    # Convert predicted boxes to DataFrame
    px = pd.DataFrame(boxes).astype("float")

    # Iterate through each bounding box
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        # Check the class name and append the bounding box to respective lists
        if 'bsufcomply' in c:
            bounding_list.append([x1, y1, x2, y2])
        elif 'bsumcomply' in c:
            bounding_list1.append([x1, y1, x2, y2])
        elif 'bsunoncomply' in c:
            bounding_list2.append([x1, y1, x2, y2])

    # Draw area of interest polygon on frame
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 255, 0), 1)
    count = len(area_c)
    # Display count of detected areas
    cv2.putText(frame, str(count), (50, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    # Display frame
    cv2.imshow("AREA", frame)

    # Break loop on ESC key press
    if cv2.waitKey(0) & 0xFF == 27:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
