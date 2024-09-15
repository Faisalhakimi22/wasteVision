import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
import warnings

# Suppress deprecation warnings from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module=".*common")

# Set the cache directory dynamically based on environment
torch.hub.set_dir(os.path.join(os.getcwd(), 'cache'))

@st.cache_data
def load_model():
    # Assuming 'best.pt' is in the same directory as the app
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

st.title("Objects Detector :tea: :coffee:")

# Button for image/video upload
upload = st.file_uploader(label="Upload Image or Video:", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"])

# Function to resize image to 640x640
def resize_image(img_array, size=(640, 640)):
    return cv2.resize(img_array, size)

# Function to make predictions
def make_prediction(img):
    # Resize image to 640x640 before making predictions
    img_resized = resize_image(img)
    results = model(img_resized)  # Perform detection
    return results

# Function to create image with bounding boxes
def create_image_with_bboxes(img_array, results):
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]  # Labels and coordinates
    n = len(labels)
    img_height, img_width, _ = img_array.shape

    # Loop through detections and draw bounding boxes
    for i in range(n):
        row = coords[i]
        if row[4] >= 0.3:  # If confidence score is above threshold (e.g., 0.3)
            x1, y1, x2, y2 = int(row[0] * img_width), int(row[1] * img_height), int(row[2] * img_width), int(row[3] * img_height)
            img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            label = model.names[int(labels[i])]  # Get the label
            img_array = cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return img_array

# Function to test camera indices
def test_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        st.write(f"Error: Unable to access camera at index {index}.")
        return
    st.write(f"Camera at index {index} is working.")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Unable to capture frame.")
            break
        cv2.imshow(f'Camera Feed {index}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if upload is not None:
    if upload.type.startswith("image"):
        img = Image.open(upload)
        img_array = np.array(img)

        # Display uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Run YOLOv5 prediction
        prediction = make_prediction(img_array)
        img_with_bbox = create_image_with_bboxes(img_array, prediction)

        # Display image with bounding boxes
        st.image(img_with_bbox, caption="Detected Objects", use_column_width=True)

    elif upload.type.startswith("video"):
        st.video(upload)

        # OpenCV to read video
        tfile = open("temp_video.mp4", "wb")
        tfile.write(upload.read())
        cap = cv2.VideoCapture("temp_video.mp4")

        stframe = st.empty()  # Placeholder for the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from the video.")
                break

            # Run YOLOv5 prediction
            results = make_prediction(frame)
            frame_with_bbox = create_image_with_bboxes(frame, results)

            stframe.image(frame_with_bbox, channels="BGR")

        cap.release()

# Button for real-time object detection
if st.button("Start Real-Time Detection"):
    stframe = st.empty()

    # Test different camera indices
    for index in range(5):
        st.write(f"Testing camera index {index}...")
        test_camera(index)

    # Use default camera index (0)
    cap = cv2.VideoCapture(0)  # Adjust the index if necessary

    if not cap.isOpened():
        st.error("Unable to access the camera. Please check your camera settings and permissions.")
        st.write(f"Debug: Camera access failed for index 0.")
    else:
        st.write("Camera is accessing...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from the camera.")
                st.write(f"Debug: Unable to grab frame from the camera.")
                break

            # Run YOLOv5 prediction
            results = make_prediction(frame)
            frame_with_bbox = create_image_with_bboxes(frame, results)

            stframe.image(frame_with_bbox, channels="BGR")

        cap.release()
