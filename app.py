
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
/* Main App Background Styling */
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://as2.ftcdn.net/v2/jpg/00/67/08/17/1000_F_67081713_yoB2gKhW150YEYMLKxP9VgceF1OGAQLy.jpg");
    background-position: top;
    background-repeat: no-repeat;
    background-attachment: local; /* Ensure the background scrolls with content */
     /* Optional: Ensures the image covers the background space */
}}

@media (max-width: 768px) {{
    [data-testid="stAppViewContainer"] > .main {{
        background-size: contain; /* Display the background at actual size on mobile */
        background-position: top;
        background-attachment: local; /* Allow background to scroll with content */
    }}
}}

/* Sidebar background */
[data-testid="stSidebar"] > div:first-child {{
    background-image: url("https://images.pexels.com/photos/7829475/pexels-photo-7829475.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
    background-position: center;
    background-repeat: no-repeat;
}}

@media (max-width: 768px) {{
    [data-testid="stSidebar"] > div:first-child {{
        background-attachment: scroll;
    }}
}}

/* Transparent header to keep the background visible */
[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

/* Toolbar adjustment */
[data-testid="stToolbar"] {{
    right: 2rem;
}}

/* Logo and Title Styling */
.logo {{
    position: absolute;
    top: 120px;
    left: 50%;
    transform: translateX(-50%);
    width: 160px;
    z-index: 1;
}}

.title {{
    position: absolute;
    top: 250px; /* Web mode: restore the original spacing */
    left: 50%;
    transform: translateX(-50%);
    font-weight: bold;
    font-size: 2em;
    color: black;
    z-index: 1;
}}

@media (max-width: 768px) {{
    .logo {{
        top: 20px;
        width: 150px;
    }}
    .title {{
        top: 150px; /* Mobile mode: adjust title position */
        font-size: 1.2em;
        text-align: center;
    }}

    /* Description Styling for Mobile View */
    .description {{
        margin-top: 10px !important; /* Mobile view: reduce space between description and title */
        position: relative;
        opacity: 0.9;
        color: black;
        font-size: 1.2em;
        text-align: center;
        max-width: 80%;
        margin-left: auto;
        margin-right: auto;
    }}
}}

/* Content Container Styling */
.container {{
    margin-top: 220px; /* Web mode: restore the original margin */
}}

.description {{
    margin-top: 100px; /* Web mode: keep original spacing */
    opacity: 0.9;
    color: black;
    font-size: 1.2em;
    text-align: center;
    max-width: 80%;
    margin-left: auto;
    margin-right: auto;
}}

.image-container {{
    opacity: 0.9;
}}

.video-container {{
    opacity: 0.9;
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
st.sidebar.markdown('<h3 style="color:white;">About</h3>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="color:white;">Welcome to EcoVision! Our app leverages advanced computer vision to automate and streamline the recycling process. With intelligent sorting bins and waste identification, EcoVision makes recycling easier and more efficient, helping you contribute to a cleaner, sustainable future. Join us in transforming waste management and making a positive impact on the environment!</p>',
    unsafe_allow_html=True
)

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

    # Set a base font size relative to the image size
    base_font_scale = 0.0015 * min(img_height, img_width)  # Adjust this scaling factor as needed
    base_thickness = max(1, int(0.005 * min(img_height, img_width)))  # Adjust thickness dynamically
    
    # Loop through detections and draw bounding boxes
    for i in range(n):
        row = coords[i]
        if row[4] >= 0.3:  # Confidence threshold
            x1, y1, x2, y2 = int(row[0] * img_width), int(row[1] * img_height), int(row[2] * img_width), int(row[3] * img_height)
            img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            
            # Dynamically adjust font size and thickness for label readability
            label = model.names[int(labels[i])]  # Get the label
            font_scale = base_font_scale  # Scale font size dynamically
            font_thickness = base_thickness  # Scale font thickness dynamically
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Define padding and background rectangle for label
            padding = int(0.01 * min(img_height, img_width))  # Adjust padding dynamically
            label_bg_start = (x1, y1 - text_size[1] - 2 * padding)  # Top-left corner of label background
            label_bg_end = (x1 + text_size[0] + 2 * padding, y1)  # Bottom-right corner of label background
            
            # Draw background rectangle behind label (use darker color for better contrast)
            cv2.rectangle(img_array, label_bg_start, label_bg_end, (50, 50, 50), -1)  # Dark gray background
            
            # Position the label text slightly below the top of the background rectangle
            text_position = (x1 + padding, y1 - padding)
            
            # Add the label text in white for contrast
            img_array = cv2.putText(img_array, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

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
# WebRTC configuration
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun1.l.google.com:19302"], "username": "", "credential": ""}]})

# Streamlit WebRTC component for real-time camera feed
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, rtc_configuration=RTC_CONFIG, media_stream_constraints={"video": True, "audio": False})

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

