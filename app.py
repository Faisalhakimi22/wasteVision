import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
import warnings
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import base64

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

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Custom CSS for styling with background images
img_sidebar = encode_image_to_base64("new.jpeg")  # Sidebar background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://images.unsplash.com/photo-1517319725296-466c84bd7d54?q=80&w=1470&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("https://images.unsplash.com/photo-1650112274147-03a2dba421c8?q=80&w=1374&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-size: cover; /* Fit the sidebar */
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
    right: 2rem;
}}

.logo {{
    position: absolute;
    top: 0px; /* Adjust to avoid overlap */
    left: 50%;
    transform: translateX(-50%);
    width: 150px; /* Adjust size as needed */
    z-index: 1; /* Ensure logo is above background */
}}

.title {{
    position: absolute;
    top: 100px; /* Adjust position as needed */
    left: 50%;
    transform: translateX(-50%);
    font-weight: bold;
    font-size: 2em; /* Adjust size as needed */
    color: black; /* Adjust color as needed */
    z-index: 1;
}}

@media (max-width: 768px) {{
    .logo {{
        top: 20px; /* Adjust for mobile view */
        width: 120px; /* Optional: Adjust size for smaller screens */
    }}
    .title {{
        top: 80px; /* Adjust for mobile view */
        font-size: 1.0em; /* Adjust size for smaller screens */
    }}
}}

.container {{
    margin-top: 150px; /* Adjust based on logo and title size */
}}

.image-container {{
    opacity: 0.3; /* Transparent image container */
}}

.video-container {{
    opacity: 0.3; /* Transparent video container */
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Add the logo to the foreground
logo_base64 = encode_image_to_base64("logo1.png")
st.markdown(f'<img src="data:image/png;base64,{logo_base64}" class="logo" alt="Logo">', unsafe_allow_html=True)
st.markdown('<div class="title">Smart Object Detector</div>', unsafe_allow_html=True)

# Content container
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<div class="description">Unveil the power of AI in recognizing and analyzing objects. Upload your media or use real-time detection to see the magic in action!</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Add an image or logo to the sidebar
st.sidebar.image('logo1.png', width=150)  # Add your logo image here
st.sidebar.markdown('### About')
st.sidebar.markdown('Welcome to EcoVision! Our app leverages advanced computer vision to automate and streamline the recycling process. With intelligent sorting bins and waste identification, EcoVision makes recycling easier and more efficient, helping you contribute to a cleaner, sustainable future. Join us in transforming waste management and making a positive impact on the environment!')

#Button for image/video upload
upload = st.file_uploader(label="Upload Image or Video:", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"], help="Upload an image or video file for detection.", label_visibility="visible")

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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model  # Load the YOLOv5 model

    def transform(self, frame):
        img_array = frame.to_ndarray(format="bgr24")
        results = make_prediction(img_array)
        img_with_bbox = create_image_with_bboxes(img_array, results)
        return img_with_bbox

# WebRTC configuration
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Streamlit WebRTC component for real-time camera feed
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, rtc_configuration=RTC_CONFIG)

if upload is not None:
    if upload.type.startswith("image"):
        img = Image.open(upload)
        img_array = np.array(img)

        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Run YOLOv5 prediction
        prediction = make_prediction(img_array)
        img_with_bbox = create_image_with_bboxes(img_array, prediction)

        # Display image with bounding boxes
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img_with_bbox, caption="Detected Objects", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif upload.type.startswith("video"):
        st.video(upload)

        # OpenCV to read video
        tfile = open("temp_video.mp4", "wb")
        tfile.write(upload.read())
        cap = cv2.VideoCapture("temp_video.mp4")

        stframe = st.empty()  # Placeholder for video frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = make_prediction(frame_rgb)
            frame_with_bbox = create_image_with_bboxes(frame_rgb, results)
            stframe.image(frame_with_bbox, channels="RGB", use_column_width=True)

        cap.release()
        os.remove("temp_video.mp4")  # Clean up temporary file

    else:
        st.warning("Unsupported file type. Please upload an image or video.")
