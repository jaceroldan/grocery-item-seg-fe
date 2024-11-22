# pip install opencv-python numpy gradio requests

import cv2
import requests
import time
import numpy as np
import gradio as gr

# Define the server endpoint
url = "http://202.92.159.242:5678/process_frame"

def process_frame_webcam(input_image):
    # Convert input image (NumPy array) to JPEG format
    _, img_encoded = cv2.imencode('.jpg', input_image)
    files = {'file': img_encoded.tobytes()}

    # Send frame to the server
    response = requests.post(url, files=files)

    # Decode the returned frame
    segmented_frame = cv2.imdecode(
        np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR
    )

    return segmented_frame

# Gradio interface setup
def gradio_infer(input_image):
    start_time = time.time()

    # Process the input frame
    segmented_frame = process_frame_webcam(input_image)

    # Calculate FPS
    fps = 1 / (time.time() - start_time)

    # Display FPS on the frame
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

    return segmented_frame[:, :, ::-1]  # Convert BGR to RGB for Gradio display

# Define Gradio interface
webcam_interface = gr.Interface(
    fn=gradio_infer,
    inputs=gr.Image(label="input", sources="webcam", streaming=True),
    outputs=gr.Image(),
    live=True,
    title="Real-Time Grocery Segmentation",
    description="A YOLO-based model for grocery segmentation using your webcam."
)

if __name__ == "__main__":
    webcam_interface.launch()
