import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import warnings
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import base64
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import json
import time
from collections import Counter, defaultdict
import threading

# Configure page
st.set_page_config(
    page_title="EcoVision AI - Smart Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*common")
torch.hub.set_dir(os.path.join(os.getcwd(), 'cache'))

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'realtime_stats' not in st.session_state:
    st.session_state.realtime_stats = defaultdict(int)

@st.cache_resource
def load_model():
    """Load YOLOv5 model with caching"""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Fallback to pretrained model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()
        return model

# Custom CSS for modern styling
def get_custom_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Main container styling */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    /* Cards styling */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    
    .detection-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .main-container {
            margin: 0.5rem;
            padding: 1rem;
        }
    }
    </style>
    """

def create_header():
    """Create modern header with logo and title"""
    st.markdown("""
    <div class="header-container fade-in">
        <div class="main-title">🔍 EcoVision AI</div>
        <div class="subtitle">Advanced Object Detection & Analysis Platform</div>
        <div style="font-size: 0.9rem; opacity: 0.8;">
            Powered by YOLOv5 • Real-time Detection • Smart Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_metrics_dashboard():
    """Create metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card pulse">
            <h3 style="margin: 0; font-size: 2rem;">{st.session_state.total_detections}</h3>
            <p style="margin: 0.5rem 0 0 0;">Total Detections</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_objects = len(set([item['class'] for item in st.session_state.detection_history]))
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{unique_objects}</h3>
            <p style="margin: 0.5rem 0 0 0;">Unique Objects</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_confidence = np.mean([item['confidence'] for item in st.session_state.detection_history]) if st.session_state.detection_history else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{avg_confidence:.1%}</h3>
            <p style="margin: 0.5rem 0 0 0;">Avg Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sessions_today = len([item for item in st.session_state.detection_history 
                            if item.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 2rem;">{sessions_today}</h3>
            <p style="margin: 0.5rem 0 0 0;">Today's Sessions</p>
        </div>
        """, unsafe_allow_html=True)

def create_detection_analytics():
    """Create detection analytics charts"""
    if not st.session_state.detection_history:
        st.info("📊 Analytics will appear here after your first detection!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Object distribution pie chart
        class_counts = Counter([item['class'] for item in st.session_state.detection_history])
        
        fig_pie = px.pie(
            values=list(class_counts.values()),
            names=list(class_counts.keys()),
            title="Object Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence distribution histogram
        confidences = [item['confidence'] for item in st.session_state.detection_history]
        
        fig_hist = px.histogram(
            x=confidences,
            title="Confidence Score Distribution",
            nbins=20,
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12),
            xaxis_title="Confidence Score",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def resize_image(img_array, size=(640, 640)):
    """Resize image maintaining aspect ratio"""
    h, w = img_array.shape[:2]
    if h == w:
        return cv2.resize(img_array, size)
    
    # Maintain aspect ratio
    if h > w:
        new_h, new_w = size[0], int(w * size[0] / h)
    else:
        new_h, new_w = int(h * size[1] / w), size[1]
    
    resized = cv2.resize(img_array, (new_w, new_h))
    
    # Pad to square
    delta_w = size[1] - new_w
    delta_h = size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def make_prediction(img, model, conf_threshold=0.3):
    """Make prediction with the model"""
    img_resized = resize_image(img)
    results = model(img_resized)
    return results, conf_threshold

def create_enhanced_image_with_bboxes(img_array, results, conf_threshold=0.3):
    """Create image with enhanced bounding boxes and labels"""
    if len(results.xyxyn[0]) == 0:
        return img_array, []
    
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    img_height, img_width, _ = img_array.shape
    
    detections = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i in range(n):
        row = coords[i]
        confidence = float(row[4])
        
        if confidence >= conf_threshold:
            x1 = int(row[0] * img_width)
            y1 = int(row[1] * img_height)
            x2 = int(row[2] * img_width)
            y2 = int(row[3] * img_height)
            
            label_idx = int(labels[i])
            label = results.names[label_idx]
            color = colors[label_idx % len(colors)]
            
            # Draw enhanced bounding box
            thickness = max(2, int(0.003 * min(img_height, img_width)))
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label with confidence
            label_text = f"{label} ({confidence:.2%})"
            font_scale = max(0.4, 0.001 * min(img_height, img_width))
            font_thickness = max(1, int(0.002 * min(img_height, img_width)))
            
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Label background
            label_bg = (x1, y1 - text_size[1] - 10, x1 + text_size[0] + 10, y1)
            cv2.rectangle(img_array, (label_bg[0], label_bg[1]), (label_bg[2], label_bg[3]), color, -1)
            
            # Label text
            cv2.putText(img_array, label_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
            # Store detection info
            detection_info = {
                'class': label,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            detections.append(detection_info)
    
    return img_array, detections

def update_detection_history(detections):
    """Update detection history in session state"""
    st.session_state.detection_history.extend(detections)
    st.session_state.total_detections += len(detections)
    
    # Keep only last 1000 detections to manage memory
    if len(st.session_state.detection_history) > 1000:
        st.session_state.detection_history = st.session_state.detection_history[-1000:]

class AdvancedVideoTransformer(VideoTransformerBase):
    def __init__(self, model, conf_threshold=0.3):
        self.model = model
        self.conf_threshold = conf_threshold
        self.frame_count = 0
        self.fps_counter = 0
        self.start_time = time.time()
    
    def transform(self, frame):
        img_array = frame.to_ndarray(format="bgr24")
        results, _ = make_prediction(img_array, self.model, self.conf_threshold)
        img_with_bbox, detections = create_enhanced_image_with_bboxes(img_array, results, self.conf_threshold)
        
        # Update realtime stats
        for detection in detections:
            st.session_state.realtime_stats[detection['class']] += 1
        
        # Add FPS counter
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.start_time >= 1.0:
            self.fps_counter = self.frame_count
            self.frame_count = 0
            self.start_time = current_time
        
        # Add FPS text to frame
        cv2.putText(img_with_bbox, f"FPS: {self.fps_counter}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img_with_bbox

def main():
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Create header
    create_header()
    
    # Load model
    with st.spinner("🚀 Loading AI Model..."):
        model = load_model()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Model settings
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
        
        # Detection mode
        detection_mode = st.radio(
            "Detection Mode",
            ["📸 Image/Video Upload", "📹 Real-time Camera", "📊 Analytics Dashboard"]
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("## 📈 Quick Stats")
        if st.session_state.detection_history:
            recent_classes = [item['class'] for item in st.session_state.detection_history[-10:]]
            most_common = Counter(recent_classes).most_common(3)
            
            for class_name, count in most_common:
                st.markdown(f"**{class_name}**: {count}")
        else:
            st.info("No detections yet")
        
        st.markdown("---")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.session_state.total_detections = 0
            st.session_state.realtime_stats = defaultdict(int)
            st.success("History cleared!")
        
        # Export data
        if st.session_state.detection_history and st.button("📥 Export Data"):
            df = pd.DataFrame(st.session_state.detection_history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Main content based on mode
    if detection_mode == "📊 Analytics Dashboard":
        st.markdown("## 📊 Analytics Dashboard")
        create_metrics_dashboard()
        st.markdown("---")
        create_detection_analytics()
        
        # Recent detections table
        if st.session_state.detection_history:
            st.markdown("## 🕒 Recent Detections")
            recent_df = pd.DataFrame(st.session_state.detection_history[-20:])
            st.dataframe(recent_df, use_container_width=True)
    
    elif detection_mode == "📹 Real-time Camera":
        st.markdown("## 📹 Real-time Object Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # WebRTC configuration
            RTC_CONFIG = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun1.l.google.com:19302"]}]
            })
            
            webrtc_streamer(
                key="advanced_detection",
                video_transformer_factory=lambda: AdvancedVideoTransformer(model, conf_threshold),
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
        
        with col2:
            st.markdown("### 🎯 Live Stats")
            if st.session_state.realtime_stats:
                for class_name, count in st.session_state.realtime_stats.items():
                    st.metric(class_name, count)
            else:
                st.info("Start detection to see live stats")
    
    else:  # Image/Video Upload mode
        st.markdown("## 📸 Image & Video Detection")
        
        # File uploader with enhanced styling
        uploaded_file = st.file_uploader(
            "Choose an image or video file",
            type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "webm"],
            help="Upload an image or video file for AI-powered object detection"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.type
            
            if file_type.startswith("image"):
                # Image processing
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📷 Original Image")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Detection Results")
                    
                    # Process image
                    img_array = np.array(image)
                    if img_array.shape[2] == 4:  # RGBA to RGB
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    
                    with st.spinner("🔍 Analyzing image..."):
                        results, _ = make_prediction(img_array, model, conf_threshold)
                        img_with_bbox, detections = create_enhanced_image_with_bboxes(
                            img_array, results, conf_threshold
                        )
                    
                    st.image(img_with_bbox, use_container_width=True)
                    
                    # Update history
                    if detections:
                        update_detection_history(detections)
                        
                        # Show detection summary
                        st.markdown("### 📋 Detection Summary")
                        for i, detection in enumerate(detections, 1):
                            st.markdown(f"""
                            <div class="detection-card">
                                <strong>{i}. {detection['class']}</strong><br>
                                Confidence: {detection['confidence']:.2%}<br>
                                Time: {detection['timestamp']}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No objects detected. Try adjusting the confidence threshold.")
            
            elif file_type.startswith("video"):
                # Video processing
                st.markdown("### 🎬 Video Processing")
                
                # Save uploaded video temporarily
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_file.read())
                
                # Video preview
                st.video("temp_video.mp4")
                
                # Process video button
                if st.button("🚀 Process Video"):
                    cap = cv2.VideoCapture("temp_video.mp4")
                    
                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    frame_placeholder = st.empty()
                    
                    all_detections = []
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process every 5th frame for performance
                        if frame_count % 5 == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results, _ = make_prediction(frame_rgb, model, conf_threshold)
                            frame_with_bbox, detections = create_enhanced_image_with_bboxes(
                                frame_rgb, results, conf_threshold
                            )
                            
                            # Display current frame
                            frame_placeholder.image(frame_with_bbox, channels="RGB", use_container_width=True)
                            
                            # Collect detections
                            all_detections.extend(detections)
                        
                        # Update progress
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    cap.release()
                    os.remove("temp_video.mp4")
                    
                    # Update history and show results
                    if all_detections:
                        update_detection_history(all_detections)
                        
                        st.success(f"✅ Video processed! Found {len(all_detections)} objects.")
                        
                        # Video summary
                        st.markdown("### 📊 Video Analysis Summary")
                        video_stats = Counter([d['class'] for d in all_detections])
                        
                        cols = st.columns(min(len(video_stats), 4))
                        for i, (class_name, count) in enumerate(video_stats.most_common(4)):
                            with cols[i]:
                                st.metric(class_name, count)
                    else:
                        st.warning("No objects detected in the video.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; opacity: 0.7; padding: 1rem;'>"
        "Made with ❤️ using Streamlit & YOLOv5 | EcoVision AI © 2024"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
