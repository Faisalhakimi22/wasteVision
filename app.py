import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import pathlib
import os

# Ensure compatibility with Windows file system in case of cross-platform usage
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load your custom YOLOv5 model
@st.cache_data
def load_model():
    # Use a relative path for deployment
    model_path = os.path.join(os.getcwd(), 'yolov5', 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

model = load_model()

st.title("Objects Detector :tea: :coffee:")

# Button for image/video upload
upload = st.file_uploader(label="Upload Image or Video:", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"])

# Function to make predictions
def make_prediction(img):
    results = model(img)  # Perform detection
    return results

# Function to create image with bounding boxes
def create_image_with_bboxes(img_array, results):
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]  # Labels and coordinates
    n = len(labels)
    img_height, img_width, _ = img_array.shape

    for i in range(n):
        row = coords[i]
        if row[4] >= 0.3:  # Confidence threshold
            x1, y1, x2, y2 = int(row[0] * img_width), int(row[1] * img_height), int(row[2] * img_width), int(row[3] * img_height)
            img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            label = model.names[int(labels[i])]
            img_array = cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return img_array

if upload is not None:
    if upload.type.startswith("image"):
        img = Image.open(upload)
        img_array = np.array(img)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        prediction = make_prediction(img_array)
        img_with_bbox = create_image_with_bboxes(img_array, prediction)
        st.image(img_with_bbox, caption="Detected Objects", use_column_width=True)

    elif upload.type.startswith("video"):
        st.video(upload)
        tfile = open("temp_video.mp4", "wb")
        tfile.write(upload.read())
        cap = cv2.VideoCapture("temp_video.mp4")

        stframe = st.empty()  # Placeholder for video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = make_prediction(frame)
            frame_with_bbox = create_image_with_bboxes(frame, results)
            stframe.image(frame_with_bbox, channels="BGR")

        cap.release()

if st.button("Start Real-Time Detection"):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)  # Webcam (0 is default camera)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = make_prediction(frame)
        frame_with_bbox = create_image_with_bboxes(frame, results)
        stframe.image(frame_with_bbox, channels="BGR")

    cap.release()
