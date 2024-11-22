# pip install opencv-python numpy

import cv2
import requests
import time
import numpy as np

# Define the server endpoint
url = "http://202.92.159.242:5678/process_frame"

cap = cv2.VideoCapture(0)  # Open webcam

# Initialize variables for FPS calculation
fps = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Start timing for the current frame
    start_time = time.time()
    
    # Encode the frame as JPEG for transmission
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'file': img_encoded.tobytes()}

    # Send frame to the server
    response = requests.post(url, files=files)

    # Receive and display the segmented frame
    segmented_frame = cv2.imdecode(
        np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
    )

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - prev_time)
    prev_time = end_time

    # Display FPS on the segmented frame
    cv2.putText(
        segmented_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Show the frame
    cv2.imshow("Segmented Output", segmented_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
