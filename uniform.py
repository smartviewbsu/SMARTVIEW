import os
import cv2
import time
import queue
import cvzone
import threading
import numpy as np
import pandas as pd
from excel import *
from ultralytics import YOLO
from object_tracker import *
from datetime import datetime


def process_bbox(area_polygon, input_frame, bounding_boxes, object_ids):
    """
    Process bounding boxes within the specified area polygon.

    Args:
        area_polygon (list): List of points defining the area polygon.
        input_frame (numpy.ndarray): Input frame.
        bounding_boxes (list): List of bounding boxes in the format [xmin, ymin, xmax, ymax, obj_id].
        object_ids (list): List of object IDs.

    Returns:
        input_frame (numpy.ndarray): Modified input frame with bounding boxes and object IDs drawn.
        object_count (int): Number of objects detected in the area.
    """
    for bbox in bounding_boxes:
        xmin, ymin, xmax, ymax, obj_id = bbox
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        result = cv2.pointPolygonTest(np.array(area_polygon, np.int32), (cx, cy), False)

        # Check if the centroid of the bounding box lies within the area polygon
        if result >= 0:
            # Draw bounding box and object ID on the frame
            cv2.rectangle(input_frame, (xmin, ymin), (xmax, ymax), (0, 0, 128), 3)
            cvzone.putTextRect(input_frame, f'{obj_id}', (xmin, ymin), scale=1, thickness=2,
                               colorT=(255, 255, 255), colorR=(0, 0, 128))

            # Append object ID to the list if not already present
            if obj_id not in object_ids:
                object_ids.append(obj_id)

    # Calculate the number of objects detected in the area
    object_count = len(object_ids)

    return input_frame, object_count


def process_video(stream_path, class_path, model, bsufcomply, bsumcomply, bsunoncomply, output_file_name):
    """
    Process a video stream to detect classes of objects within frames, create a new video with annotations, and continuously save the frames to a video file.

    Args:
        stream_path (str): Path to the input video stream.
        class_path (str): Path to the file containing classes to detect.
        model: The object detection model to use.
        bsufcomply: Argument description for bsufcomply.
        bsumcomply: Argument description for bsumcomply.
        bsunoncomply: Argument description for bsunoncomply.
        output_file_name (str): Name of the output video file.

    Returns:
        None
    """

    # Open the video stream
    cap = cv2.VideoCapture(stream_path)

    # Define codec for video saving
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Get video properties
    fps = 20.0
    frame_width = 640
    frame_height = 640

    # Define output VideoWriter object
    out = cv2.VideoWriter(output_file_name, fourcc, fps, (frame_width, frame_height))

    # Read class data from file
    my_file = open(class_path, "r")
    data = my_file.read()
    class_list = data.split("\n")

    # Initialize lists to store compliance statuses
    bsufcomply_list = []
    bsumcomply_list = []
    bsunoncomply_list = []

    # Define a queue to store frames for saving
    frame_queue = queue.Queue()

    def save_frame(frame_queue_func):
        while True:
            frame_func = frame_queue_func.get()
            if frame_func is None:
                break
            out.write(frame_func)

    # Create thread for saving frames
    save_thread = threading.Thread(target=save_frame, args=(frame_queue,))
    save_thread.start()

    # Define area of interest
    area = [(0, 310), (0, 370), (628, 390), (615, 375)]

    # Define time intervals
    six_to_seven = False  # 6:00 am to 7:00 am
    seven_to_eight = False  # 7:00 am to 8:00 am
    eight_to_nine = False  # 8:00 am to 9:00 am
    nine_to_ten = False  # 9:00 am to 10:00 am
    ten_to_eleven = False  # 10:00 am to 11:00 am
    eleven_to_twelve = False  # 11:00 am to 12:00 pm
    twelve_to_thirteen = False  # 12:00 am to 1:00 pm
    thirteen_to_fourteen = False  # 1:00 pm to 2:00 pm
    fourteen_to_fifteen = False  # 2:00 pm to 3:00 pm
    fifteen_to_sixteen = False  # 3:00 pm to 4:00 pm
    sixteen_to_seventeen = False  # 4:00 pm to 5:00 pm
    seventeen_to_eighteen = False  # 5:00 pm to 6:00 pm
    eighteen_to_nineteen = False  # 6:00 pm to 7:00 pm
    nineteen = False  # 7:00 pm
    nineteen_one = False  # 7:01 pm
    nineteen_two = False  # 7:02 pm

    while True:
        # Read frame from the video stream
        ret, frame = cap.read()

        try:
            # Check if frame reading was successful
            if not ret:
                raise Exception("Error reading frames from the live stream")
        except Exception as e:
            # Handle exceptions
            print(f"An error occurred: {str(e)}")
            break

        # Resize frame
        frame = cv2.resize(frame, (640, 640))

        # Predict objects in the frame using the provided model
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        # Plot annotated frame with detected objects
        annotated_frame = results[0].plot()

        # Initialize lists to store bounding boxes for each class
        bounding_list = []
        bounding_list1 = []
        bounding_list2 = []

        # Iterate through detected objects
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            d = int(row[5])
            c = class_list[d]

            # Classify objects into respective compliance categories
            if 'bsufcomply' in c:
                bounding_list.append([x1, y1, x2, y2])
            elif 'bsumcomply' in c:
                bounding_list1.append([x1, y1, x2, y2])
            elif 'bsunoncomply' in c:
                bounding_list2.append([x1, y1, x2, y2])

        # Update bounding box indices for each compliance category
        bbox_idx = bsufcomply.update(bounding_list)
        bbox1_idx = bsumcomply.update(bounding_list1)
        bbox2_idx = bsunoncomply.update(bounding_list2)

        # Process and annotate bounding boxes within specified area
        annotated_frame, bsufcomply_count = process_bbox(area, annotated_frame, bbox_idx, bsufcomply_list)
        annotated_frame, bsumcomply_count = process_bbox(area, annotated_frame, bbox1_idx, bsumcomply_list)
        annotated_frame, bsunoncomply_count = process_bbox(area, annotated_frame, bbox2_idx, bsunoncomply_list)

        # Calculate total counts of compliant and non-compliant objects
        compliant_count = bsufcomply_count + bsumcomply_count
        non_compliant_count = bsunoncomply_count

        # Draw a polygon around the specified area
        cv2.polylines(annotated_frame, [np.array(area, np.int32)], True, (0, 255, 0), 1)

        # Display counts of compliant and non-compliant objects on the annotated frame
        cvzone.putTextRect(annotated_frame, f'BSUFCOMPLY: {bsufcomply_count}', (30, 600),
                           scale=1, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 128),
                           border=1, colorB=(0, 255, 255))
        cvzone.putTextRect(annotated_frame, f'BSUMCOMPLY: {bsumcomply_count}', (250, 600),
                           scale=1, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 128),
                           border=1, colorB=(0, 255, 255))
        cvzone.putTextRect(annotated_frame, f'BSUNONCOMPLY: {bsunoncomply_count}', (450, 600),
                           scale=1, thickness=2, colorT=(255, 255, 255), colorR=(0, 0, 128),
                           border=1, colorB=(0, 255, 255))

        # Get the current time
        current_time = datetime.now().time()

        """
        This code segment implements a time-based scheduler to perform tasks at specific intervals.
        It first checks if the current time falls within the operational time range of 05:00 to 20:00.
        Within this range, it further examines the hour, minute, and second to determine which task to execute.

        Tasks involve data extraction for each one-hour interval from 6 AM to 7 PM.
        Each task is executed in a separate thread to enable concurrent processing.
        After processing data for an interval, relevant lists are cleared to prepare for the next hour.

        At the end of the day, after the last hour interval (19:00 to 20:00), additional tasks are performed.
        These tasks include clearing data from a Google sheet and resetting flags for the next cycle.
        """

        if datetime.strptime("05:00", "%H:%M").time() <= current_time <= datetime.strptime("20:00", "%H:%M").time():
            if current_time.hour == 6 and current_time.minute == 59 and current_time.second == 59 and not six_to_seven:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C4", "Sheet1!D4",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                six_to_seven = True
                nineteen_two = False
            elif current_time.hour == 7 and current_time.minute == 59 and current_time.second == 59 and not seven_to_eight:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C5", "Sheet1!D5",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                seven_to_eight = True
            elif current_time.hour == 8 and current_time.minute == 59 and current_time.second == 59 and not eight_to_nine:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C6", "Sheet1!D6",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                eight_to_nine = True
            elif current_time.hour == 9 and current_time.minute == 59 and current_time.second == 59 and not nine_to_ten:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C7", "Sheet1!D7",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                nine_to_ten = True
            elif current_time.hour == 10 and current_time.minute == 59 and current_time.second == 59 and not ten_to_eleven:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C8", "Sheet1!D8",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                ten_to_eleven = True
            elif current_time.hour == 11 and current_time.minute == 59 and current_time.second == 59 and not eleven_to_twelve:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C9", "Sheet1!D9",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                eleven_to_twelve = True
            elif current_time.hour == 12 and current_time.minute == 59 and current_time.second == 59 and not twelve_to_thirteen:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C10", "Sheet1!D10",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                twelve_to_thirteen = True
            elif current_time.hour == 13 and current_time.minute == 59 and current_time.second == 59 and not thirteen_to_fourteen:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C11", "Sheet1!D11",
                                                                      compliant_count,
                                                                      non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                thirteen_to_fourteen = True
            elif current_time.hour == 14 and current_time.minute == 59 and current_time.second == 59 and not fourteen_to_fifteen:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C12", "Sheet1!D12",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                fourteen_to_fifteen = True
            elif current_time.hour == 15 and current_time.minute == 59 and current_time.second == 59 and not fifteen_to_sixteen:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C13", "Sheet1!D13",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                fifteen_to_sixteen = True
            elif current_time.hour == 16 and current_time.minute == 59 and current_time.second == 59 and not sixteen_to_seventeen:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C14", "Sheet1!D14",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                sixteen_to_seventeen = True
            elif current_time.hour == 17 and current_time.minute == 59 and current_time.second == 59 and not seventeen_to_eighteen:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C15", "Sheet1!D15",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                seventeen_to_eighteen = True
            elif current_time.hour == 18 and current_time.minute == 59 and current_time.second == 59 and not eighteen_to_nineteen:
                google_sheet_raw_data_thread = threading.Thread(target=google_sheet_raw_data,
                                                                args=("Sheet1!C1", "Sheet1!C16", "Sheet1!D16",
                                                                      compliant_count, non_compliant_count))
                google_sheet_raw_data_thread.start()
                bsufcomply_list.clear()
                bsumcomply_list.clear()
                bsunoncomply_list.clear()
                eighteen_to_nineteen = True
            elif current_time.hour == 19 and current_time.minute == 0 and current_time.second == 30 and not nineteen:
                print("Total")
                google_sheet_total_data_thread = threading.Thread(target=google_sheet_total_data)
                google_sheet_total_data_thread.start()
                nineteen = True
            elif current_time.hour == 19 and current_time.minute == 1 and current_time.second == 0 and not nineteen_one:
                print("Download")
                download_google_sheet_and_save_thread = threading.Thread(target=download_google_sheet_and_save)
                download_google_sheet_and_save_thread.start()
                nineteen_one = True
            elif current_time.hour == 19 and current_time.minute == 2 and current_time.second == 0 and not nineteen_two:
                print("Clear")
                clear_google_sheet_data_thread = threading.Thread(target=clear_google_sheet_data)
                clear_google_sheet_data_thread.start()
                nineteen_two = True
                six_to_seven = False
                seven_to_eight = False
                eight_to_nine = False
                nine_to_ten = False
                ten_to_eleven = False
                eleven_to_twelve = False
                twelve_to_thirteen = False
                thirteen_to_fourteen = False
                fourteen_to_fifteen = False
                fifteen_to_sixteen = False
                sixteen_to_seventeen = False
                seventeen_to_eighteen = False
                eighteen_to_nineteen = False
                nineteen = False
                nineteen_one = False
            else:
                pass

        frame_queue.put(annotated_frame)

        # Display the annotated frame in a window titled "SMARTVIEW"
        cv2.imshow("SMARTVIEW", annotated_frame)

        # Wait for a key press event, if the pressed key is ESC (key code 27), break the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    out.release()

    # Signal the save thread to stop
    frame_queue.put(None)
    save_thread.join()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


def main():
    """
    Main function to process video stream using YOLO object detection model and track objects.
    """

    # Path to the video stream
    stream_path = 'haircolorlorenze.mp4'

    # Path to the file containing class labels
    class_path = 'smartview_classes.txt'

    # Initialize YOLO object detection model with pre-trained weights
    model = YOLO('best-l.pt')

    # Initialize object trackers for different categories
    bsufcomply_tracker = Object_Tracker()
    bsumcomply_tracker = Object_Tracker()
    bsunoncomply_tracker = Object_Tracker()

    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Get current time
    current_time = time.strftime("%H-%M-%S")

    # Define output file name
    output_file_name = f"SMARTVIEW-{current_time}-{current_date}.mp4"

    # Create a thread to process the video stream
    video_thread = threading.Thread(target=process_video, args=(
        stream_path, class_path, model, bsufcomply_tracker, bsumcomply_tracker, bsunoncomply_tracker, output_file_name))

    # Start the video processing thread
    video_thread.start()


if __name__ == "__main__":
    # Call the main function when the script is executed
    main()
