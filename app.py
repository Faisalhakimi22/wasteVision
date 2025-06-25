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

# Configure page with modern settings
st.set_page_config(
    page_title="EcoVision AI - Advanced Object Detection Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.ecovision-ai.com/help',
        'Report a bug': 'https://www.ecovision-ai.com/bug',
        'About': "EcoVision AI - Next-generation object detection powered by advanced AI"
    }
)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*common")
torch.hub.set_dir(os.path.join(os.getcwd(), 'cache'))

# Initialize session state with enhanced tracking
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'realtime_stats' not in st.session_state:
    st.session_state.realtime_stats = defaultdict(int)
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = datetime.now()
if 'processing_time_history' not in st.session_state:
    st.session_state.processing_time_history = []

@st.cache_resource
def load_model():
    """Load YOLOv5 model with enhanced error handling"""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Custom model loading failed: {str(e)}")
        # Fallback to pretrained model
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.eval()
            st.info("✅ Using pretrained YOLOv5s model as fallback")
            return model
        except Exception as e2:
            st.error(f"❌ Failed to load any model: {str(e2)}")
            return None

def load_logo():
    """Load and encode logo image"""
    try:
        with open("logo1.png", "rb") as file:
            return base64.b64encode(file.read()).decode()
    except FileNotFoundError:
        st.warning("⚠️ Logo file 'logo1.png' not found in current directory")
        return None

# Modern professional CSS with advanced styling
def get_advanced_css():
    logo_base64 = load_logo()
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height: 60px; margin-right: 15px; vertical-align: middle;">' if logo_base64 else ''
    
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Root variables for consistent theming */
    :root {{
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --glass-border: rgba(255, 255, 255, 0.2);
        --shadow-light: 0 8px 32px rgba(31, 38, 135, 0.37);
        --shadow-heavy: 0 20px 60px rgba(0, 0, 0, 0.1);
        --border-radius: 20px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    /* Global font and base styling */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
    }}
    
    /* Advanced gradient background with animation */
    .stApp {{
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
    }}
    
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* Glassmorphism main container */
    .main-container {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1rem;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--glass-border);
        position: relative;
        overflow: hidden;
    }}
    
    .main-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--primary-gradient);
    }}
    
    /* Modern header with logo integration */
    .header-container {{
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        color: white;
        box-shadow: var(--shadow-heavy);
        position: relative;
        overflow: hidden;
    }}
    
    .header-container::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        animation: float 20s infinite linear;
        pointer-events: none;
    }}
    
    @keyframes float {{
        0% {{ transform: translate(-50%, -50%) rotate(0deg); }}
        100% {{ transform: translate(-50%, -50%) rotate(360deg); }}
    }}
    
    .logo-title-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        position: relative;
        z-index: 2;
    }}
    
    .main-title {{
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
        margin: 0;
        letter-spacing: -0.02em;
    }}
    
    .subtitle {{
        font-size: 1.25rem;
        font-weight: 400;
        opacity: 0.95;
        margin-bottom: 1rem;
        position: relative;
        z-index: 2;
    }}
    
    .tech-stack {{
        font-size: 0.9rem;
        opacity: 0.8;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 2;
    }}
    
    .tech-badge {{
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        transition: var(--transition);
    }}
    
    .tech-badge:hover {{
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.3);
    }}
    
    /* Advanced metric cards with enhanced animations */
    .metric-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        backdrop-filter: blur(20px);
        padding: 2rem 1.5rem;
        border-radius: var(--border-radius);
        text-align: center;
        box-shadow: var(--shadow-light);
        margin-bottom: 1rem;
        border: 1px solid var(--glass-border);
        position: relative;
        overflow: hidden;
        transition: var(--transition);
        cursor: pointer;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: var(--transition);
    }}
    
    .metric-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }}
    
    .metric-card:hover::before {{
        left: 100%;
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
        font-family: 'JetBrains Mono', monospace;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }}
    
    .metric-icon {{
        font-size: 2rem;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }}
    
    /* Enhanced detection cards */
    .detection-card {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: var(--shadow-light);
        margin: 1rem 0;
        border-left: 4px solid transparent;
        border-image: var(--primary-gradient) 1;
        position: relative;
        transition: var(--transition);
    }}
    
    .detection-card:hover {{
        transform: translateX(5px);
        box-shadow: var(--shadow-heavy);
    }}
    
    .detection-card::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--primary-gradient);
        border-radius: 0 2px 2px 0;
    }}
    
    /* Modern button styling */
    .stButton > button {{
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: var(--transition);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: var(--transition);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:active {{
        transform: translateY(-1px);
    }}
    
    /* Enhanced sidebar styling */
    .css-1d391kg {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
    }}
    
    /* Modern progress bar */
    .stProgress > div > div > div > div {{
        background: var(--primary-gradient);
        border-radius: 10px;
    }}
    
    /* File uploader enhancement */
    .stFileUploader > div > div {{
        background: rgba(255, 255, 255, 0.9);
        border: 2px dashed #667eea;
        border-radius: var(--border-radius);
        padding: 3rem 2rem;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }}
    
    .stFileUploader > div > div:hover {{
        border-color: #764ba2;
        background: rgba(255, 255, 255, 0.95);
        transform: scale(1.02);
    }}
    
    /* Advanced animations */
    .fade-in {{
        animation: fadeIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .slide-in-left {{
        animation: slideInLeft 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-50px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    .slide-in-right {{
        animation: slideInRight 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    @keyframes slideInRight {{
        from {{ opacity: 0; transform: translateX(50px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    .pulse-glow {{
        animation: pulseGlow 2s infinite;
    }}
    
    @keyframes pulseGlow {{
        0%, 100% {{ box-shadow: 0 0 20px rgba(102, 126, 234, 0.4); }}
        50% {{ box-shadow: 0 0 40px rgba(102, 126, 234, 0.8); }}
    }}
    
    /* Status indicators */
    .status-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.25rem;
    }}
    
    .status-online {{
        background: rgba(34, 197, 94, 0.1);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }}
    
    .status-processing {{
        background: rgba(251, 191, 36, 0.1);
        color: #d97706;
        border: 1px solid rgba(251, 191, 36, 0.2);
    }}
    
    .status-offline {{
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }}
    
    /* Loading spinner */
    .loading-spinner {{
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-top: 2px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Responsive design */
    @media (max-width: 768px) {{
        .main-title {{
            font-size: 2rem;
        }}
        .main-container {{
            margin: 0.5rem;
            padding: 1rem;
        }}
        .header-container {{
            padding: 2rem 1rem;
        }}
        .logo-title-container {{
            flex-direction: column;
            gap: 1rem;
        }}
        .tech-stack {{
            flex-direction: column;
            gap: 0.5rem;
        }}
    }}
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {{
        .main-container {{
            background: rgba(30, 30, 30, 0.95);
            color: #ffffff;
        }}
        .metric-card {{
            background: rgba(50, 50, 50, 0.9);
            color: #ffffff;
        }}
        .detection-card {{
            background: rgba(40, 40, 40, 0.95);
            color: #ffffff;
        }}
    }}
    </style>
    """

def create_modern_header():
    """Create modern header with logo and enhanced styling"""
    logo_base64 = load_logo()
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="height: 60px; margin-right: 15px; filter: drop-shadow(0 0 10px rgba(255,255,255,0.3));">' if logo_base64 else '<i class="fas fa-brain" style="font-size: 3rem; margin-right: 15px;"></i>'
    
    st.markdown(f"""
    <div class="header-container fade-in">
        <div class="logo-title-container">
            {logo_html}
            <div class="main-title">EcoVision AI</div>
        </div>
        <div class="subtitle">
            <i class="fas fa-robot"></i> Next-Generation Object Detection & Analysis Platform
        </div>
        <div class="tech-stack">
            <div class="tech-badge">
                <i class="fas fa-brain"></i> YOLOv5
            </div>
            <div class="tech-badge">
                <i class="fas fa-bolt"></i> Real-time Processing
            </div>
            <div class="tech-badge">
                <i class="fas fa-chart-line"></i> Smart Analytics
            </div>
            <div class="tech-badge">
                <i class="fas fa-cloud"></i> Cloud-Ready
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_advanced_metrics_dashboard():
    """Create advanced metrics dashboard with modern styling"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card pulse-glow slide-in-left">
            <div class="metric-icon"><i class="fas fa-search"></i></div>
            <div class="metric-value">{st.session_state.total_detections:,}</div>
            <div class="metric-label">Total Detections</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_objects = len(set([item['class'] for item in st.session_state.detection_history]))
        st.markdown(f"""
        <div class="metric-card slide-in-left" style="animation-delay: 0.1s;">
            <div class="metric-icon"><i class="fas fa-cubes"></i></div>
            <div class="metric-value">{unique_objects}</div>
            <div class="metric-label">Unique Objects</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_confidence = np.mean([item['confidence'] for item in st.session_state.detection_history]) if st.session_state.detection_history else 0
        st.markdown(f"""
        <div class="metric-card slide-in-right" style="animation-delay: 0.2s;">
            <div class="metric-icon"><i class="fas fa-bullseye"></i></div>
            <div class="metric-value">{avg_confidence:.1%}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        session_duration = (datetime.now() - st.session_state.session_start_time).total_seconds() / 60
        st.markdown(f"""
        <div class="metric-card slide-in-right" style="animation-delay: 0.3s;">
            <div class="metric-icon"><i class="fas fa-clock"></i></div>
            <div class="metric-value">{session_duration:.1f}m</div>
            <div class="metric-label">Session Time</div>
        </div>
        """, unsafe_allow_html=True)

def create_advanced_analytics():
    """Create advanced analytics with modern charts"""
    if not st.session_state.detection_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255,255,255,0.1); border-radius: 20px; backdrop-filter: blur(10px);">
            <i class="fas fa-chart-bar" style="font-size: 4rem; color: #667eea; margin-bottom: 1rem;"></i>
            <h3 style="color: #667eea; margin-bottom: 1rem;">Analytics Dashboard</h3>
            <p style="color: #64748b;">Advanced analytics will appear here after your first detection session</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced object distribution chart
        class_counts = Counter([item['class'] for item in st.session_state.detection_history])
        
        fig_pie = px.pie(
            values=list(class_counts.values()),
            names=list(class_counts.keys()),
            title="<b>Object Distribution Analysis</b>",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12),
            title_x=0.5,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Enhanced confidence distribution
        confidences = [item['confidence'] for item in st.session_state.detection_history]
        
        fig_hist = px.histogram(
            x=confidences,
            title="<b>Confidence Score Distribution</b>",
            nbins=25,
            color_discrete_sequence=['#667eea'],
            opacity=0.8
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12),
            title_x=0.5,
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            bargap=0.1
        )
        fig_hist.add_vline(
            x=np.mean(confidences),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {np.mean(confidences):.2%}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Timeline analysis
    if len(st.session_state.detection_history) > 1:
        st.markdown("### 📈 Detection Timeline")
        
        # Create timeline data
        timeline_data = []
        for i, detection in enumerate(st.session_state.detection_history):
            timeline_data.append({
                'Index': i + 1,
                'Class': detection['class'],
                'Confidence': detection['confidence'],
                'Timestamp': detection.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        fig_timeline = px.scatter(
            df_timeline,
            x='Index',
            y='Confidence',
            color='Class',
            title="<b>Detection Confidence Over Time</b>",
            hover_data=['Timestamp']
        )
        fig_timeline.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12),
            title_x=0.5
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

def resize_image(img_array, size=(640, 640)):
    """Resize image maintaining aspect ratio with enhanced processing"""
    h, w = img_array.shape[:2]
    if h == w:
        return cv2.resize(img_array, size, interpolation=cv2.INTER_LANCZOS4)
    
    # Maintain aspect ratio
    if h > w:
        new_h, new_w = size[0], int(w * size[0] / h)
    else:
        new_h, new_w = int(h * size[1] / w), size[1]
    
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Pad to square with smart padding
    delta_w = size[1] - new_w
    delta_h = size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Use edge replication instead of black padding
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)

def make_prediction(img, model, conf_threshold=0.3):
    """Enhanced prediction function with timing"""
    start_time = time.time()
    img_resized = resize_image(img)
    results = model(img_resized)
    processing_time = time.time() - start_time
    
    # Store processing time for analytics
    st.session_state.processing_time_history.append(processing_time)
    if len(st.session_state.processing_time_history) > 100:
        st.session_state.processing_time_history = st.session_state.processing_time_history[-100:]
    
    return results, conf_threshold, processing_time

def create_enhanced_image_with_bboxes(img_array, results, conf_threshold=0.3):
    """Create image with professional bounding boxes and labels"""
    if len(results.xyxyn[0]) == 0:
        return img_array, []
    
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    img_height, img_width, _ = img_array.shape
    
    detections = []
    # Professional color palette
    colors = [
        (102, 126, 234),  # Primary blue
        (118, 75, 162),   # Primary purple
        (240, 147, 251),  # Light pink
        (245, 87, 108),   # Coral
        (79, 172, 254),   # Sky blue
        (56, 249, 215),   # Turquoise
        (67, 233, 123),   # Green
        (255, 183, 77)    # Orange
    ]
    
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
            
            # Draw enhanced bounding box with gradient effect
            thickness = max(3, int(0.004 * min(img_height, img_width)))
            
            # Main bounding box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, thickness)
            
            # Corner decorations for modern look
            corner_length = min(20, (x2-x1)//6, (y2-y1)//6)
            corner_thickness = thickness + 1
            
            # Top-left corner
            cv2.line(img_array, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
            cv2.line(img_array, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
            
            # Top-right corner
            cv2.line(img_array, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
            cv2.line(img_array, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
            
            # Bottom-left corner
            cv2.line(img_array, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
            cv2.line(img_array, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
            
            # Bottom-right corner
            cv2.line(img_array, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
            cv2.line(img_array, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
            
            # Enhanced label with professional styling
            label_text = f"{label} {confidence:.1%}"
            font_scale = max(0.5, 0.0012 * min(img_height, img_width))
            font_thickness = max(1, int(0.003 * min(img_height, img_width)))
            
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)[0]
            
            # Modern label background with rounded corners effect
            label_bg_height = text_size[1] + 16
            label_bg_width = text_size[0] + 20
            
            # Create gradient background for label
            overlay = img_array.copy()
            cv2.rectangle(overlay, (x1, y1 - label_bg_height), (x1 + label_bg_width, y1), color, -1)
            cv2.addWeighted(overlay, 0.8, img_array, 0.2, 0, img_array)
            
            # Label border
            cv2.rectangle(img_array, (x1, y1 - label_bg_height), (x1 + label_bg_width, y1), color, 2)
            
            # Label text with shadow effect
            text_x, text_y = x1 + 10, y1 - 8
            
            # Text shadow
            cv2.putText(img_array, label_text, (text_x + 1, text_y + 1), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), font_thickness + 1)
            
            # Main text
            cv2.putText(img_array, label_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), font_thickness)
            
            # Store enhanced detection info
            detection_info = {
                'class': label,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'area': (x2 - x1) * (y2 - y1),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
            }
            detections.append(detection_info)
    
    return img_array, detections

def update_detection_history(detections):
    """Enhanced detection history management"""
    st.session_state.detection_history.extend(detections)
    st.session_state.total_detections += len(detections)
    
    # Maintain optimal history size
    if len(st.session_state.detection_history) > 1000:
        st.session_state.detection_history = st.session_state.detection_history[-1000:]

class AdvancedVideoTransformer(VideoTransformerBase):
    """Enhanced video transformer with advanced features"""
    def __init__(self, model, conf_threshold=0.3):
        self.model = model
        self.conf_threshold = conf_threshold
        self.frame_count = 0
        self.fps_counter = 0
        self.start_time = time.time()
        self.detection_buffer = []
        self.processing_times = []
    
    def transform(self, frame):
        img_array = frame.to_ndarray(format="bgr24")
        
        # Process frame
        start_time = time.time()
        results, _, processing_time = make_prediction(img_array, self.model, self.conf_threshold)
        img_with_bbox, detections = create_enhanced_image_with_bboxes(img_array, results, self.conf_threshold)
        
        # Update realtime stats
        for detection in detections:
            st.session_state.realtime_stats[detection['class']] += 1
        
        # FPS calculation
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.start_time >= 1.0:
            self.fps_counter = self.frame_count
            self.frame_count = 0
            self.start_time = current_time
        
        # Add professional overlay with system stats
        self.add_professional_overlay(img_with_bbox, len(detections), processing_time)
        
        return img_with_bbox
    
    def add_professional_overlay(self, img, detection_count, processing_time):
        """Add professional overlay with system information"""
        h, w = img.shape[:2]
        
        # Create semi-transparent overlay panel
        overlay = img.copy()
        panel_height = 80
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Add system information
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # FPS
        cv2.putText(img, f"FPS: {self.fps_counter}", (10, 25), font, font_scale, (0, 255, 0), font_thickness)
        
        # Detection count
        cv2.putText(img, f"Detections: {detection_count}", (10, 50), font, font_scale, (102, 126, 234), font_thickness)
        
        # Processing time
        cv2.putText(img, f"Process: {processing_time*1000:.1f}ms", (200, 25), font, font_scale, (240, 147, 251), font_thickness)
        
        # Model status
        cv2.putText(img, "EcoVision AI", (w-150, 25), font, font_scale, (255, 255, 255), font_thickness)

def create_status_panel():
    """Create system status panel"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_processing_time = np.mean(st.session_state.processing_time_history) if st.session_state.processing_time_history else 0
        status_class = "status-online" if avg_processing_time < 0.5 else "status-processing" if avg_processing_time < 1.0 else "status-offline"
        st.markdown(f"""
        <div class="status-indicator {status_class}">
            <i class="fas fa-microchip"></i>
            Processing: {avg_processing_time*1000:.0f}ms
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        model_status = "online" if 'model' in locals() else "loading"
        st.markdown(f"""
        <div class="status-indicator status-{model_status}">
            <i class="fas fa-brain"></i>
            Model: {'Ready' if model_status == 'online' else 'Loading'}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        memory_usage = len(st.session_state.detection_history)
        memory_status = "online" if memory_usage < 800 else "processing"
        st.markdown(f"""
        <div class="status-indicator status-{memory_status}">
            <i class="fas fa-memory"></i>
            Memory: {memory_usage}/1000
        </div>
        """, unsafe_allow_html=True)

def main():
    """Enhanced main application"""
    # Apply advanced CSS
    st.markdown(get_advanced_css(), unsafe_allow_html=True)
    
    # Create modern header
    create_modern_header()
    
    # System status panel
    create_status_panel()
    
    # Load model with enhanced loading
    with st.spinner("🧠 Initializing AI Neural Networks..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        model = load_model()
        progress_bar.empty()
    
    if model is None:
        st.error("❌ Failed to load model. Please check your setup.")
        return
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Advanced Configuration")
        
        # Model settings with enhanced controls
        st.markdown("### 🎯 Detection Parameters")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05, 
                                  help="Higher values = more confident detections")
        
        iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05,
                                 help="Non-maximum suppression threshold")
        
        max_det = st.slider("Max Detections", 1, 100, 50, 1,
                           help="Maximum number of detections per image")
        
        # Detection mode with enhanced options
        st.markdown("### 🔧 Operation Mode")
        detection_mode = st.radio(
            "Select Mode",
            ["📸 Smart Upload", "📹 Live Stream", "📊 Analytics Hub", "🔬 Batch Processing"]
        )
        
        st.markdown("---")
        
        # Performance monitoring
        st.markdown("### 📈 Performance Monitor")
        if st.session_state.processing_time_history:
            avg_time = np.mean(st.session_state.processing_time_history)
            st.metric("Avg Processing Time", f"{avg_time*1000:.1f}ms")
            
            # Performance chart
            fig_perf = px.line(
                y=st.session_state.processing_time_history[-20:],
                title="Processing Time Trend",
                labels={'y': 'Time (s)', 'x': 'Recent Frames'}
            )
            fig_perf.update_layout(
                height=200,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10)
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        
        st.markdown("---")
        
        # Advanced controls
        st.markdown("### 🛠️ Advanced Tools")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.detection_history = []
                st.session_state.total_detections = 0
                st.session_state.realtime_stats = defaultdict(int)
                st.session_state.processing_time_history = []
                st.success("✅ All data cleared!")
        
        with col2:
            if st.button("📊 Export", use_container_width=True):
                if st.session_state.detection_history:
                    df = pd.DataFrame(st.session_state.detection_history)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"ecovision_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No data to export")
    
    # Main content area
    if detection_mode == "📊 Analytics Hub":
        st.markdown("## 📊 Advanced Analytics Dashboard")
        create_advanced_metrics_dashboard()
        st.markdown("---")
        create_advanced_analytics()
        
        # Detailed analytics
        if st.session_state.detection_history:
            st.markdown("### 📋 Detection History")
            
            # Create enhanced dataframe
            df = pd.DataFrame(st.session_state.detection_history)
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            with col1:
                class_filter = st.multiselect("Filter by Class", df['class'].unique())
            with col2:
                min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0)
            with col3:
                max_results = st.slider("Max Results", 10, 100, 50)
            
            # Apply filters
            filtered_df = df.copy()
            if class_filter:
                filtered_df = filtered_df[filtered_df['class'].isin(class_filter)]
            filtered_df = filtered_df[filtered_df['confidence'] >= min_conf]
            filtered_df = filtered_df.tail(max_results)
            
            # Display enhanced table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Detection confidence score",
                        min_value=0,
                        max_value=1,
                        format="%.1%"
                    ),
                    "timestamp": st.column_config.DatetimeColumn(
                        "Timestamp",
                        help="When the detection occurred"
                    )
                }
            )
    
    elif detection_mode == "📹 Live Stream":
        st.markdown("## 📹 Real-time Object Detection Stream")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Enhanced WebRTC configuration
            RTC_CONFIG = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]}
                ]
            })
            
            webrtc_streamer(
                key="ecovision_stream",
                video_transformer_factory=lambda: AdvancedVideoTransformer(model, conf_threshold),
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={
                    "video": {"width": 1280, "height": 720, "frameRate": 30},
                    "audio": False
                },
                async_processing=True
            )
        
        with col2:
            st.markdown("### 🎯 Live Detection Stats")
            
            # Real-time metrics
            if st.session_state.realtime_stats:
                for class_name, count in sorted(st.session_state.realtime_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.markdown(f"""
                    <div class="detection-card" style="margin: 0.5rem 0; padding: 1rem;">
                        <strong>{class_name}</strong><br>
                        <span style="font-size: 1.5rem; color: #667eea;">{count}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🔄 Start detection to see live stats")
            
            # Stream controls
            st.markdown("### 🎮 Stream Controls")
            
            if st.button("📸 Capture Frame", use_container_width=True):
                st.info("Frame capture functionality would be implemented here")
            
            if st.button("🎬 Record Stream", use_container_width=True):
                st.info("Stream recording functionality would be implemented here")
    
    elif detection_mode == "🔬 Batch Processing":
        st.markdown("## 🔬 Batch Processing Mode")
        
        st.info("📁 Upload multiple files for batch processing")
        
        uploaded_files = st.file_uploader(
            "Select multiple images or videos",
            type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "webm"],
            accept_multiple_files=True,
            help="Upload multiple files for efficient batch processing"
        )
        
        if uploaded_files:
            st.markdown(f"### 📊 Batch Queue ({len(uploaded_files)} files)")
            
            # Batch processing controls
            col1, col2, col3 = st.columns(3)
            with col1:
                process_all = st.button("🚀 Process All", use_container_width=True)
            with col2:
                save_results = st.checkbox("💾 Save Results")
            with col3:
                show_progress = st.checkbox("📊 Show Progress", value=True)
            
            if process_all:
                batch_results = []
                
                if show_progress:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    if show_progress:
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        status_text.text(f"Processing {file.name}...")
                    
                    # Process file
                    if file.type.startswith("image"):
                        image = Image.open(file)
                        img_array = np.array(image)
                        if img_array.shape[2] == 4:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                        
                        results, _, proc_time = make_prediction(img_array, model, conf_threshold)
                        _, detections = create_enhanced_image_with_bboxes(img_array, results, conf_threshold)
                        
                        batch_results.append({
                            'file': file.name,
                            'type': 'image',
                            'detections': len(detections),
                            'processing_time': proc_time,
                            'objects': [d['class'] for d in detections]
                        })
                
                # Show batch results
                if batch_results:
                    st.markdown("### 📋 Batch Processing Results")
                    
                    results_df = pd.DataFrame(batch_results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Batch statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Files", len(batch_results))
                    with col2:
                        st.metric("Total Detections", sum(r['detections'] for r in batch_results))
                    with col3:
                        st.metric("Avg Processing Time", f"{np.mean([r['processing_time'] for r in batch_results])*1000:.1f}ms")
    
    else:  # Smart Upload mode
        st.markdown("## 📸 Smart Upload & Analysis")
        
        # Enhanced file uploader
        uploaded_file = st.file_uploader(
            "Upload Image or Video",
            type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "webm"],
            help="Supported formats: PNG, JPG, JPEG, MP4, AVI, MOV, WEBM"
        )
        
        if uploaded_file is not None:
            file_type = uploaded_file.type
            
            if file_type.startswith("image"):
                # Enhanced image processing
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 📷 Original Image")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)
                    
                    # Image metadata
                    st.markdown("**Image Information:**")
                    st.markdown(f"- **Size:** {image.size[0]} × {image.size[1]} pixels")
                    st.markdown(f"- **Format:** {image.format}")
                    st.markdown(f"- **Mode:** {image.mode}")
                
                with col2:
                    st.markdown("### 🎯 AI Detection Results")
                    
                    # Process image
                    img_array = np.array(image)
                    if img_array.shape[2] == 4:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    
                    with st.spinner("🔍 AI Analysis in progress..."):
                        results, _, processing_time = make_prediction(img_array, model, conf_threshold)
                        img_with_bbox, detections = create_enhanced_image_with_bboxes(
                            img_array, results, conf_threshold
                        )
                    
                    st.image(img_with_bbox, use_container_width=True)
                    
                    # Processing metrics
                    st.markdown(f"**⚡ Processing Time:** {processing_time*1000:.1f}ms")
                    
                    # Update history
                    if detections:
                        update_detection_history(detections)
                        
                        # Enhanced detection summary
                        st.markdown("### 📋 Detection Summary")
                        
                        for i, detection in enumerate(detections, 1):
                            confidence_color = "#22c55e" if detection['confidence'] > 0.8 else "#f59e0b" if detection['confidence'] > 0.5 else "#ef4444"
                            
                            st.markdown(f"""
                            <div class="detection-card fade-in" style="animation-delay: {i*0.1}s;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong style="font-size: 1.1rem;">🎯 {detection['class']}</strong><br>
                                        <span style="color: {confidence_color}; font-weight: 600;">
                                            Confidence: {detection['confidence']:.1%}
                                        </span><br>
                                        <small style="color: #64748b;">
                                            📅 {detection['timestamp']}<br>
                                            📏 Size: {detection['area']:,} pixels
                                        </small>
                                    </div>
                                    <div style="font-size: 2rem;">
                                        {'🟢' if detection['confidence'] > 0.8 else '🟡' if detection['confidence'] > 0.5 else '🔴'}
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("🔍 No objects detected. Try adjusting the confidence threshold in the sidebar.")
    
    # Enhanced footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; backdrop-filter: blur(10px); margin-top: 2rem;'>
            <div style='font-size: 1.2rem; margin-bottom: 1rem;'>
                <i class='fas fa-heart' style='color: #ef4444;'></i> 
                <strong>EcoVision AI</strong> - Powered by Advanced Neural Networks
            </div>
            <div style='font-size: 0.9rem; opacity: 0.8;'>
                <i class='fas fa-code'></i> Built with Streamlit & YOLOv5 | 
                <i class='fas fa-copyright'></i> 2024 EcoVision Technologies | 
                <i class='fas fa-globe'></i> Next-Gen AI Solutions
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
