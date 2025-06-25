import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
import warnings
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import base64
import time
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import pandas as pd

# Suppress deprecation warnings from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module=".*common")

# Set page configuration
st.set_page_config(
    page_title="EcoVision - Smart Object Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the cache directory dynamically based on environment
torch.hub.set_dir(os.path.join(os.getcwd(), 'cache'))

@st.cache_data
def load_model():
    # Assuming 'best.pt' is in the same directory as the app
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Initialize session state for analytics
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0

# Function to encode image to base64
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except:
        # Return a placeholder if image not found
        return ""

# Advanced CSS styling with glassmorphism and modern design
def get_advanced_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main App Background */
    [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #2D3748 0%, #1A202C 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Header Section */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 300;
        margin-bottom: 2rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    
    /* Stats Cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Upload Section */
    .upload-section {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Results Section */
    .results-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Sidebar Styling */
    .sidebar-content {
        color: white;
        padding: 1rem;
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #667eea;
    }
    
    .sidebar-text {
        color: rgba(255, 255, 255, 0.8);
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
    }
    
    .feature-description {
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.6;
    }
    
    /* Detection Info */
    .detection-info {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .app-title {
            font-size: 2.5rem;
        }
        
        .stats-container {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 4px solid #667eea;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #667eea, #764ba2);
    }
    </style>
    """

# Apply advanced CSS
st.markdown(get_advanced_css(), unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="header-container">
    <div class="logo-container">
        <!-- Logo will be inserted here if available -->
    </div>
    <h1 class="app-title">🌱 EcoVision</h1>
    <p class="app-subtitle">
        Intelligent Object Detection powered by Advanced AI - Transform your world with smart recognition technology
    </p>
</div>
""", unsafe_allow_html=True)

# Stats Section
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.total_detections}</div>
        <div class="stat-label">Total Detections</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(set([item['class'] for item in st.session_state.detection_history]))}</div>
        <div class="stat-label">Unique Objects</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(st.session_state.detection_history)}</div>
        <div class="stat-label">Sessions</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_confidence = np.mean([item['confidence'] for item in st.session_state.detection_history if st.session_state.detection_history]) if st.session_state.detection_history else 0
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{avg_confidence:.1%}</div>
        <div class="stat-label">Avg Confidence</div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with enhanced design
with st.sidebar:
    # Try to load logo
    try:
        st.image('logo1.png', width=150)
    except:
        st.markdown("### 🌱 EcoVision")
    
    st.markdown("""
    <div class="sidebar-content">
        <div class="sidebar-title">About EcoVision</div>
        <div class="sidebar-text">
            Welcome to EcoVision, where cutting-edge AI meets environmental responsibility. 
            Our advanced computer vision technology revolutionizes object detection and recycling automation.
        </div>
        
        <div class="sidebar-title">Key Features</div>
        <div class="sidebar-text">
            • Real-time object detection<br>
            • Advanced AI-powered analysis<br>
            • Smart recycling classification<br>
            • Detailed analytics dashboard<br>
            • Multi-format support (images/videos)
        </div>
        
        <div class="sidebar-title">Detection Settings</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced settings
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
    show_labels = st.checkbox("Show Labels", value=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    
    st.markdown("---")
    
    # Analytics toggle
    show_analytics = st.checkbox("Show Analytics Dashboard", value=False)

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Analytics", "🎥 Live Camera"])

with tab1:
    # Feature cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">📷</div>
            <div class="feature-title">Image Detection</div>
            <div class="feature-description">Upload images in PNG, JPG, or JPEG format for instant object detection and classification.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎬</div>
            <div class="feature-title">Video Analysis</div>
            <div class="feature-description">Process video files with frame-by-frame detection for comprehensive analysis.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-time Processing</div>
            <div class="feature-description">Advanced AI algorithms ensure fast and accurate detection with minimal latency.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Your Media")
    st.markdown("Drag and drop or click to upload an image or video file")
    
    upload = st.file_uploader(
        label="Choose file",
        type=["png", "jpg", "jpeg", "mp4", "avi", "mov"],
        help="Supported formats: PNG, JPG, JPEG, MP4, AVI, MOV",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced prediction functions
def resize_image(img_array, size=(640, 640)):
    return cv2.resize(img_array, size)

def make_prediction(img):
    img_resized = resize_image(img)
    results = model(img_resized)
    return results

def create_advanced_image_with_bboxes(img_array, results, show_labels=True, show_confidence=True, conf_threshold=0.3):
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    img_height, img_width, _ = img_array.shape
    
    # Enhanced styling
    base_font_scale = 0.0015 * min(img_height, img_width)
    base_thickness = max(2, int(0.005 * min(img_width, img_height)))
    
    detections = []
    
    for i in range(n):
        row = coords[i]
        confidence = row[4]
        if confidence >= conf_threshold:
            x1, y1, x2, y2 = int(row[0] * img_width), int(row[1] * img_height), int(row[2] * img_width), int(row[3] * img_height)
            
            # Enhanced bounding box with gradient effect
            color = (0, 255, 0) if confidence > 0.7 else (255, 165, 0) if confidence > 0.5 else (255, 0, 0)
            img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), color, base_thickness)
            
            # Add corner markers for modern look
            corner_length = 20
            cv2.line(img_array, (x1, y1), (x1 + corner_length, y1), color, base_thickness + 1)
            cv2.line(img_array, (x1, y1), (x1, y1 + corner_length), color, base_thickness + 1)
            cv2.line(img_array, (x2, y2), (x2 - corner_length, y2), color, base_thickness + 1)
            cv2.line(img_array, (x2, y2), (x2, y2 - corner_length), color, base_thickness + 1)
            
            if show_labels or show_confidence:
                label = model.names[int(labels[i])]
                text = f"{label}"
                if show_confidence:
                    text += f" ({confidence:.2f})"
                
                # Enhanced label styling
                font_scale = base_font_scale * 1.2
                font_thickness = base_thickness
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                
                padding = int(0.01 * min(img_height, img_width))
                label_bg_start = (x1, y1 - text_size[1] - 2 * padding)
                label_bg_end = (x1 + text_size[0] + 2 * padding, y1)
                
                # Gradient background
                cv2.rectangle(img_array, label_bg_start, label_bg_end, (0, 0, 0), -1)
                cv2.rectangle(img_array, (label_bg_start[0], label_bg_start[1]), 
                            (label_bg_end[0], label_bg_start[1] + 3), color, -1)
                
                text_position = (x1 + padding, y1 - padding)
                img_array = cv2.putText(img_array, text, text_position, 
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
            # Store detection info
            detections.append({
                'class': model.names[int(labels[i])],
                'confidence': float(confidence),
                'bbox': [x1, y1, x2, y2]
            })
    
    return img_array, detections

# Video transformer for real-time detection
class AdvancedVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.frame_count = 0

    def transform(self, frame):
        self.frame_count += 1
        img_array = frame.to_ndarray(format="bgr24")
        results = make_prediction(img_array)
        img_with_bbox, detections = create_advanced_image_with_bboxes(
            img_array, results, show_labels, show_confidence, confidence_threshold
        )
        
        # Update session state with detections (throttled)
        if self.frame_count % 30 == 0:  # Update every 30 frames
            st.session_state.total_detections += len(detections)
            st.session_state.detection_history.extend(detections)
        
        return img_with_bbox

# Handle file uploads
if upload is not None:
    with st.spinner("🚀 Processing your upload..."):
        progress_bar = st.progress(0)
        
        if upload.type.startswith("image"):
            progress_bar.progress(25)
            img = Image.open(upload)
            img_array = np.array(img)
            
            # Display original image
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📷 Original Image")
                st.image(img, use_column_width=True)
            
            progress_bar.progress(50)
            
            # Run prediction
            prediction = make_prediction(img_array)
            img_with_bbox, detections = create_advanced_image_with_bboxes(
                img_array, prediction, show_labels, show_confidence, confidence_threshold
            )
            
            progress_bar.progress(75)
            
            with col2:
                st.markdown("#### 🎯 Detection Results")
                st.image(img_with_bbox, use_column_width=True)
            
            # Update session state
            st.session_state.total_detections += len(detections)
            st.session_state.detection_history.extend(detections)
            
            # Display detection summary
            if detections:
                st.markdown("### 📊 Detection Summary")
                detection_df = pd.DataFrame(detections)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Objects Detected", len(detections))
                with col2:
                    st.metric("Unique Classes", len(set([d['class'] for d in detections])))
                with col3:
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                
                # Detection details
                st.markdown("#### 🔍 Detailed Results")
                for i, detection in enumerate(detections, 1):
                    st.markdown(f"""
                    <div class="detection-info">
                        <strong>Detection {i}:</strong> {detection['class']} 
                        (Confidence: {detection['confidence']:.2%})
                    </div>
                    """, unsafe_allow_html=True)
            
            progress_bar.progress(100)
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif upload.type.startswith("video"):
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.markdown("#### 🎬 Video Processing")
            st.video(upload)
            
            # Process video
            tfile = open("temp_video.mp4", "wb")
            tfile.write(upload.read())
            cap = cv2.VideoCapture("temp_video.mp4")
            
            stframe = st.empty()
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = make_prediction(frame_rgb)
                frame_with_bbox, detections = create_advanced_image_with_bboxes(
                    frame_rgb, results, show_labels, show_confidence, confidence_threshold
                )
                
                stframe.image(frame_with_bbox, channels="RGB", use_column_width=True)
                
                # Update detections (throttled for performance)
                if frame_count % 10 == 0:
                    st.session_state.total_detections += len(detections)
                    st.session_state.detection_history.extend(detections)
            
            cap.release()
            os.remove("temp_video.mp4")
            st.markdown('</div>', unsafe_allow_html=True)
            
        progress_bar.empty()

with tab2:
    st.markdown("### 📊 Analytics Dashboard")
    
    if st.session_state.detection_history:
        # Create analytics visualizations
        df = pd.DataFrame(st.session_state.detection_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution
            class_counts = df['class'].value_counts()
            fig_pie = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Object Class Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_hist = px.histogram(
                df, x='confidence',
                title="Confidence Score Distribution",
                nbins=20,
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Detection timeline
        st.markdown("#### 📈 Detection Timeline")
        timeline_data = df.groupby('class').size().reset_index(name='count')
        fig_bar = px.bar(
            timeline_data, x='class', y='count',
            title="Objects Detected by Class",
            color='count',
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.info("📊 No detection data available yet. Upload some images or videos to see analytics!")

with tab3:
    st.markdown("### 🎥 Live Camera Detection")
    st.markdown("Experience real-time object detection using your device camera")
    
    # WebRTC configuration
    RTC_CONFIG = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun1.l.google.com:19302"]}]
    })
    
    # Camera settings
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("#### Camera Settings")
        camera_quality = st.selectbox("Quality", ["Standard", "High", "Ultra"])
        fps_limit = st.slider("FPS Limit", 1, 30, 15)
    
    with col1:
        webrtc_streamer(
            key="advanced_detection",
            video_transformer_factory=AdvancedVideoTransformer,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": {
                    "width": 640 if camera_quality == "Standard" else 1280 if camera_quality == "High" else 1920,
                    "height": 480 if camera_quality == "Standard" else 720 if camera_quality == "High" else 1080,
                    "frameRate": fps_limit
                },
                "audio": False
            }
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255, 255, 255, 0.6); padding: 2rem;">
    <p>🌱 <strong>EcoVision</strong> - Powered by Advanced AI Technology</p>
    <p>Making the world smarter, one detection at a time.</p>
</div>
""", unsafe_allow_html=True)
