import streamlit as st
import subprocess
import tempfile
import os
import base64
import cv2
import numpy as np
from PIL import Image
import warnings
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Suppress deprecation warnings from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module=".*common")

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
    background-attachment: scroll; /* Ensure background scrolls with content */
}}

/* Sidebar background */
[data-testid="stSidebar"] > div:first-child {{
    background-image: url("https://images.pexels.com/photos/7829475/pexels-photo-7829475.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: scroll; /* Ensure background scrolls with content */
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
    top: 250px;
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
        top: 150px;
        font-size: 1.2em;
        text-align: center;
    }}

    /* Description Styling for Mobile View */
    .description {{
        margin-top: 10px !important;
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
    margin-top: 220px;
}}

.description {{
    margin-top: 100px;
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
st.sidebar.markdown('<p style="color:white;">Welcome to EcoVision! Our app leverages advanced computer vision to automate and streamline the recycling process. With intelligent sorting bins and waste identification, EcoVision makes recycling easier and more efficient, helping you contribute to a cleaner, sustainable future. Join us in transforming waste management and making a positive impact on the environment!</p>', unsafe_allow_html=True)

# Button for image/video upload
upload = st.file_uploader(label="Upload Image or Video:", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"], help="Upload an image or video file for detection.", label_visibility="visible")

# Function to create a temporary file and call detect.py
def run_detect_py(image_path):
    # Path to YOLO detect.py script
    detect_script_path = 'detect.py'  # Adjust if needed
    
    # Path to your trained model
    model_path = 'best.pt'  # Adjust if needed

    # Run the detect.py script
    command = [
        'python', detect_script_path,
        '--weights', model_path,
        '--source', image_path,
        '--save-txt', '--nosave'
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Output path where the results are saved
    output_path = 'runs/detect/exp'  # Adjust if needed

    return output_path, result.stderr

# Function to display image with bounding boxes
def display_detected_image(output_path):
    output_image_path = os.path.join(output_path, "image0.jpg")  # Adjust if needed
    if os.path.exists(output_image_path):
        result_image = Image.open(output_image_path)
        return result_image
    else:
        st.error("Error: Detection results not found.")
        return None

# WebRTC configuration
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Streamlit WebRTC component for real-time camera feed
webrtc_streamer(key="example", video_transformer_factory=VideoTransformerBase, rtc_configuration=RTC_CONFIG)

if upload is not None:
    # Handle image
    if upload.type.startswith("image"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file_path = temp_file.name
            img = Image.open(upload)
            img.save(temp_file_path)
        
        # Run detection
        output_path, error = run_detect_py(temp_file_path)
        if not error:
            result_image = display_detected_image(output_path)
            if result_image:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(result_image, caption="Detected Objects", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"Error running detection: {error}")
        
        # Clean up temporary file
        os.remove(temp_file_path)

    # Handle video
    elif upload.type.startswith("video"):
        st.video(upload)

        # Save video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file_path = temp_file.name
            tfile = open(temp_file_path, "wb")
            tfile.write(upload.read())
            tfile.close()
        
        # OpenCV to read video
        cap = cv2.VideoCapture(temp_file_path)
        stframe = st.empty()  # Placeholder for video frame

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Save frame to a temporary file
            frame_temp_path = "temp_frame.jpg"
            cv2.imwrite(frame_temp_path, frame_rgb)
            
            # Run detection
            output_path, error = run_detect_py(frame_temp_path)
            if not error:
                result_image = display_detected_image(output_path)
                if result_image:
                    stframe.image(result_image, channels="RGB", use_column_width=True)
            
            # Clean up temporary frame file
            os.remove(frame_temp_path)

        cap.release()
        os.remove(temp_file_path)  # Clean up temporary video file

    else:
        st.warning("Unsupported file type. Please upload an image or video.")
