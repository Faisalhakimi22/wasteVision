def create_sidebar():
    """Create professional sidebar with enhanced controls"""
    
    # Company branding in sidebar
    logo_files = ["logo1.png", "logo.png", "company_logo.png", "sidebar_logo.png"]
    sidebar_logo = None
    
    for logo_file in logo_files:
        sidebar_logo = encode_image_to_base64(logo_file)
        if sidebar_logo:
            st.sidebar.markdown(f'<div style="text-align: center; padding: 20px 0;"><img src="data:image/png;base64,{sidebar_logo}" style="width: 120px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);" alt="Logo"></div>', unsafe_allow_html=True)
            break
    
    st.sidebar.markdown('<div style="text-align: center; color: #f1f5f9; font-size: 1.2rem; font-weight: 600; margin-bottom: 30px;">Detection Controls</div>', unsafe_allow_html=True)
    
    # Professional section headers
    st.sidebar.markdown('<div style="color: #3b82f6; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 24px 0 12px 0; border-bottom: 1px solid rgba(59, 130, 246, 0.3); padding-bottom: 4px;">🎯 MODEL SETTINGS</div>', unsafe_allow_html=True)
    
    # Model settings with professional styling
    confidence = st.sidebar.slider(
        "Confidence Threshold", 
        0.1, 1.0, 0.3, 0.05,
        help="Minimum confidence score for object detection"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold", 
        0.1, 1.0, 0.45, 0.05,
        help="Intersection over Union threshold for non-maximum suppression"
    )
    
    max_detections = st.sidebar.slider(
        "Max Detections", 
        10, 200, 100, 10,
        help="Maximum number of objects to detect per image"
    )
    
    # Image enhancement section
    st.sidebar.markdown('<div style="color: #8b5cf6; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 24px 0 12px 0; border-bottom: 1px solid rgba(139, 92, 246, 0.3); padding-bottom: 4px;">🎨 IMAGE ENHANCEMENT</div>', unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        enhance_contrast = st.sidebar.checkbox("🔆 Contrast", False)
        enhance_brightness = st.sidebar.checkbox("💡 Brightness", False)
    with col2:
        enhance_sharpness = st.sidebar.checkbox("🔍 Sharpness", False)
        auto_enhance = st.sidebar.checkbox("⚡ Auto", False)
    
    # Performance settings
    st.sidebar.markdown('<div style="color: #06b6d4; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 24px 0 12px 0; border-bottom: 1px solid rgba(6, 182, 212, 0.3); padding-bottom: 4px;">⚡ PERFORMANCE</div>', unsafe_allow_html=True)
    
    processing_speed = st.sidebar.select_slider(
        "Processing Speed",
        options=["Accurate", "Balanced", "Fast"],
        value="Balanced",
        help="Balance between accuracy and processing speed"
    )
    
    batch_processing = st.sidebar.checkbox("📦 Batch Mode", False, help="Enable batch processing for multiple files")
    
    # Update model settings
    if model:
        model.conf = confidence
        model.iou = iou_threshold
        model.max_det = max_detections
    
    # Professional divider
    st.sidebar.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(71, 85, 105, 0.5), transparent); margin: 30px 0;"></div>', unsafe_allow_html=True)
    
    # Company information section
    st.sidebar.markdown('<div style="color: #f1f5f9; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 24px 0 12px 0;">🌱 ABOUT ECOVISION</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.6; padding: 16px; background: rgba(255, 255, 255, 0.05); border-radius: 12px; border: 1px solid rgba(71, 85, 105, 0.2);">
    <strong style="color: #f1f5f9;">EcoVision AI</strong> revolutionizes environmental intelligence through cutting-edge computer vision technology.
    <br><br>
    <strong style="color: #3b82f6;">🎯 Core Features:</strong><br>
    • Real-time object detection<br>
    • Advanced analytics dashboard<br>
    • Batch processing capabilities<br>
    • Mobile-optimized interface<br>
    • Environmental impact tracking<br>
    <br>
    <strong style="color: #8b5cf6;">🚀 Enterprise Ready:</strong><br>
    Built for scale, security, and performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Version and status info
    st.sidebar.markdown('<div style="text-align: center; margin-top: 20px; padding: 12px; background: rgba(15, 23, 42, 0.6); border-radius: 8px; border: 1px solid rgba(71, 85, 105, 0.2);">', unsafe_allowimport streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import warnings
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import base64
import json
import time
from datetime import datetime
import pandas as pd
from collections import defaultdict, Counter
import threading
import queue

# Suppress deprecation warnings from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module=".*common")

# Set page configuration
st.set_page_config(
    page_title="EcoVision - Smart Object Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the cache directory dynamically based on environment
torch.hub.set_dir(os.path.join(os.getcwd(), 'cache'))

# Global variables for analytics
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'detection_stats' not in st.session_state:
    st.session_state.detection_stats = defaultdict(int)

@st.cache_resource
def load_model():
    """Load YOLOv5 model with error handling"""
    try:
        # Try loading custom model first
        if os.path.exists('best.pt'):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
        else:
            # Fallback to pre-trained model
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        model.eval()
        model.conf = 0.3  # Confidence threshold
        model.iou = 0.45   # IoU threshold for NMS
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    """Load custom class names if available"""
    try:
        if os.path.exists('classes.json'):
            with open('classes.json', 'r') as f:
                return json.load(f)
        return None
    except:
        return None

# Load model and class names
model = load_model()
custom_classes = load_class_names()

# Configuration settings
class Config:
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]

def encode_image_to_base64(image_path):
    """Encode image to base64 with error handling"""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.warning(f"Could not load image {image_path}: {e}")
    return None

def apply_custom_css():
    """Apply professional enterprise-grade CSS styling"""
    
    # Get background images if they exist
    main_bg = encode_image_to_base64("background.png") or encode_image_to_base64("background.jpg") or encode_image_to_base64("bg.png") or encode_image_to_base64("bg.jpg")
    sidebar_bg = encode_image_to_base64("sidebar_bg.png") or encode_image_to_base64("sidebar_bg.jpg") or encode_image_to_base64("new.jpeg")
    
    # Professional CSS with your custom backgrounds
    professional_css = f"""
    <style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Reset and Base Styling */
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-weight: 400;
        line-height: 1.6;
        color: #1a1a1a;
        scroll-behavior: smooth;
    }}
    
    /* Main App Container with Custom Background */
    [data-testid="stAppViewContainer"] > .main {{
        background: {'url("data:image/png;base64,' + main_bg + '")' if main_bg else 'linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #475569 75%, #64748b 100%)'};
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
        min-height: 100vh;
        position: relative;
    }}
    
    /* Dark overlay for better text readability */
    [data-testid="stAppViewContainer"] > .main::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(15, 23, 42, 0.7);
        z-index: 0;
    }}
    
    /* All content above overlay */
    [data-testid="stAppViewContainer"] > .main > * {{
        position: relative;
        z-index: 1;
    }}
    
    /* Professional Sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background: {'url("data:image/png;base64,' + sidebar_bg + '")' if sidebar_bg else 'linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #334155 100%)'};
        background-size: cover;
        background-position: center;
        border-right: 1px solid rgba(71, 85, 105, 0.3);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }}
    
    /* Sidebar overlay */
    [data-testid="stSidebar"] > div:first-child::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(15, 23, 42, 0.85);
        z-index: 0;
    }}
    
    /* Sidebar content above overlay */
    [data-testid="stSidebar"] > div:first-child > * {{
        position: relative;
        z-index: 1;
    }}
    
    /* Remove default headers */
    [data-testid="stHeader"] {{
        background: transparent;
        border-bottom: none;
    }}
    
    /* Professional Header Section */
    .header-container {{
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(71, 85, 105, 0.2);
        border-radius: 24px;
        padding: 40px;
        margin: 20px auto;
        max-width: 1200px;
        text-align: center;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.5),
            0 0 0 1px rgba(255, 255, 255, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }}
    
    .header-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}
    
    /* Logo Styling */
    .company-logo {{
        width: 80px;
        height: 80px;
        border-radius: 20px;
        margin: 0 auto 20px;
        display: block;
        box-shadow: 
            0 20px 25px -5px rgba(0, 0, 0, 0.4),
            0 10px 10px -5px rgba(0, 0, 0, 0.2);
        border: 2px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }}
    
    .company-logo:hover {{
        transform: translateY(-5px) scale(1.05);
        box-shadow: 
            0 25px 30px -5px rgba(0, 0, 0, 0.5),
            0 15px 15px -5px rgba(0, 0, 0, 0.3);
    }}
    
    /* Typography */
    .main-title {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 50%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }}
    
    .main-subtitle {{
        font-size: 1.25rem;
        font-weight: 400;
        color: #94a3b8;
        margin-bottom: 32px;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.7;
    }}
    
    /* Content Cards */
    .content-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(71, 85, 105, 0.2);
        border-radius: 16px;
        padding: 32px;
        margin: 24px 0;
        box-shadow: 
            0 20px 25px -5px rgba(0, 0, 0, 0.3),
            0 10px 10px -5px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .content-card:hover {{
        transform: translateY(-2px);
        box-shadow: 
            0 25px 30px -5px rgba(0, 0, 0, 0.4),
            0 15px 15px -5px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-color: rgba(71, 85, 105, 0.3);
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(71, 85, 105, 0.2);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 15px 25px -5px rgba(59, 130, 246, 0.2);
    }}
    
    /* Professional Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.025em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 10px 15px -3px rgba(59, 130, 246, 0.3),
            0 4px 6px -2px rgba(59, 130, 246, 0.1);
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 
            0 15px 25px -5px rgba(59, 130, 246, 0.4),
            0 8px 10px -5px rgba(59, 130, 246, 0.2);
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(71, 85, 105, 0.2);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }}
    
    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
        border-radius: 4px;
    }}
    
    /* File Uploader */
    .stFileUploader > div > div {{
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(71, 85, 105, 0.4);
        border-radius: 12px;
        transition: all 0.3s ease;
    }}
    
    .stFileUploader > div > div:hover {{
        border-color: rgba(59, 130, 246, 0.6);
        background: rgba(59, 130, 246, 0.05);
    }}
    
    /* Sidebar Text Colors */
    .stSidebar .stMarkdown {{
        color: #e2e8f0;
    }}
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {{
        color: #f1f5f9;
    }}
    
    /* Dataframes */
    .stDataFrame {{
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(71, 85, 105, 0.2);
    }}
    
    /* Metrics */
    [data-testid="metric-container"] {{
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(71, 85, 105, 0.2);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }}
    
    /* Charts */
    .stPlotlyChart {{
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(71, 85, 105, 0.2);
        padding: 16px;
    }}
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {{
        .main-title {{
            font-size: 2.5rem;
        }}
        
        .main-subtitle {{
            font-size: 1.1rem;
        }}
        
        .header-container {{
            padding: 24px;
            margin: 12px;
        }}
        
        .content-card {{
            padding: 20px;
            margin: 16px 0;
        }}
        
        .company-logo {{
            width: 60px;
            height: 60px;
        }}
    }}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(15, 23, 42, 0.3);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, #2563eb, #7c3aed);
    }}
    
    /* Success/Info/Warning Messages */
    .stSuccess {{
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 12px;
    }}
    
    .stInfo {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
    }}
    
    .stWarning {{
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
    }}
    
    /* Loading Spinner */
    .stSpinner > div {{
        border-top-color: #3b82f6 !important;
    }}
    </style>
    """
    
    st.markdown(professional_css, unsafe_allow_html=True)

def create_header():
    """Create professional header with company branding"""
    
    # Professional header container
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    
    # Company logo - try multiple possible logo file names
    logo_files = ["logo1.png", "logo.png", "company_logo.png", "brand_logo.png", "logo1.jpg", "logo.jpg"]
    logo_base64 = None
    
    for logo_file in logo_files:
        logo_base64 = encode_image_to_base64(logo_file)
        if logo_base64:
            break
    
    if logo_base64:
        st.markdown(f'<img src="data:image/png;base64,{logo_base64}" class="company-logo" alt="Company Logo">', unsafe_allow_html=True)
    else:
        # Fallback: Create a professional text logo
        st.markdown('<div style="width: 80px; height: 80px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 20px; margin: 0 auto 20px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 1.5rem;">EV</div>', unsafe_allow_html=True)
    
    # Professional title and subtitle
    st.markdown('<h1 class="main-title">EcoVision AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Enterprise-Grade Object Detection & Environmental Intelligence Platform</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_sidebar():
    """Create enhanced sidebar with controls and information"""
    st.sidebar.markdown("## 🎛️ Detection Settings")
    
    # Model settings
    confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
    max_detections = st.sidebar.slider("Max Detections", 10, 200, 100, 10)
    
    # Image enhancement options
    st.sidebar.markdown("## 🖼️ Image Enhancement")
    enhance_contrast = st.sidebar.checkbox("Enhance Contrast", False)
    enhance_brightness = st.sidebar.checkbox("Enhance Brightness", False)
    enhance_sharpness = st.sidebar.checkbox("Enhance Sharpness", False)
    
    # Update model settings
    if model:
        model.conf = confidence
        model.iou = iou_threshold
        model.max_det = max_detections
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🌱 About EcoVision")
    st.sidebar.markdown("""
    **EcoVision** leverages cutting-edge AI to revolutionize waste management and recycling.
    
    **Features:**
    - 🎯 Real-time object detection
    - 📊 Advanced analytics dashboard
    - 🔄 Batch processing
    - 📱 Mobile-optimized interface
    - 🌍 Environmental impact tracking
    """)
    
    return {
        'confidence': confidence,
        'iou_threshold': iou_threshold,
        'max_detections': max_detections,
        'enhance_contrast': enhance_contrast,
        'enhance_brightness': enhance_brightness,
        'enhance_sharpness': enhance_sharpness
    }

def enhance_image(image, settings):
    """Apply image enhancements based on settings"""
    if not any([settings['enhance_contrast'], settings['enhance_brightness'], settings['enhance_sharpness']]):
        return image
    
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    
    if settings['enhance_contrast']:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
    
    if settings['enhance_brightness']:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
    
    if settings['enhance_sharpness']:
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.3)
    
    return np.array(pil_image)

def resize_image(img_array, target_size=640):
    """Intelligent image resizing with aspect ratio preservation"""
    h, w = img_array.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create padded image
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    # Place resized image in center
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded, scale, (x_offset, y_offset)

def make_prediction(img, settings):
    """Enhanced prediction function with preprocessing"""
    if model is None:
        return None, None
    
    try:
        # Apply image enhancements
        enhanced_img = enhance_image(img, settings)
        
        # Resize image with padding
        processed_img, scale, offsets = resize_image(enhanced_img)
        
        # Run inference
        results = model(processed_img)
        
        return results, (scale, offsets)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def create_advanced_bbox_image(img_array, results, transform_params=None):
    """Create image with advanced bounding boxes and labels"""
    if results is None or len(results.xyxyn[0]) == 0:
        return img_array
    
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    img_height, img_width = img_array.shape[:2]
    
    # Calculate dynamic sizing
    base_font_scale = max(0.5, min(img_height, img_width) * 0.001)
    base_thickness = max(2, int(min(img_height, img_width) * 0.004))
    
    detection_info = []
    
    for i, row in enumerate(coords):
        confidence = float(row[4])
        
        if confidence >= model.conf:
            # Transform coordinates if needed
            x1 = int(row[0] * img_width)
            y1 = int(row[1] * img_height)
            x2 = int(row[2] * img_width)
            y2 = int(row[3] * img_height)
            
            # Get class information
            class_id = int(labels[i])
            class_name = model.names[class_id] if class_id < len(model.names) else f"Class_{class_id}"
            
            if custom_classes and class_name in custom_classes:
                display_name = custom_classes[class_name]
            else:
                display_name = class_name.replace('_', ' ').title()
            
            # Color selection
            color = Config.COLORS[class_id % len(Config.COLORS)]
            
            # Draw bounding box with rounded corners effect
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, base_thickness)
            
            # Create label with confidence
            label_text = f"{display_name} {confidence:.2f}"
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, base_thickness//2
            )
            
            # Draw label background
            label_y = y1 - text_height - 10 if y1 - text_height - 10 > 0 else y1 + text_height + 10
            cv2.rectangle(img_array, 
                         (x1, label_y - text_height - 5), 
                         (x1 + text_width + 10, label_y + 5), 
                         color, -1)
            
            # Draw label text
            cv2.putText(img_array, label_text, 
                       (x1 + 5, label_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       base_font_scale, 
                       (255, 255, 255), 
                       base_thickness//2)
            
            # Store detection info
            detection_info.append({
                'class': display_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'timestamp': datetime.now()
            })
    
    # Update global statistics
    for detection in detection_info:
        st.session_state.detection_stats[detection['class']] += 1
    
    st.session_state.detection_history.extend(detection_info)
    
    return img_array

def create_analytics_dashboard():
    """Create advanced analytics dashboard using built-in Streamlit components"""
    st.markdown("## 📊 Detection Analytics")
    
    if not st.session_state.detection_history:
        st.info("No detections yet. Upload an image or start real-time detection to see analytics.")
        return
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_detections = len(st.session_state.detection_history)
        st.metric("Total Detections", total_detections)
    
    with col2:
        unique_classes = len(st.session_state.detection_stats)
        st.metric("Unique Classes", unique_classes)
    
    with col3:
        avg_confidence = np.mean([d['confidence'] for d in st.session_state.detection_history])
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        most_common = max(st.session_state.detection_stats.items(), key=lambda x: x[1])
        st.metric("Most Detected", f"{most_common[0]} ({most_common[1]})")
    
    # Charts using Streamlit built-in components
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Detection Distribution by Class")
        # Create a DataFrame for the bar chart
        stats_df = pd.DataFrame([
            {'Class': class_name, 'Count': count, 'Percentage': f"{(count/total_detections)*100:.1f}%"} 
            for class_name, count in st.session_state.detection_stats.items()
        ])
        stats_df = stats_df.sort_values('Count', ascending=False)
        
        # Display as bar chart
        st.bar_chart(stats_df.set_index('Class')['Count'])
        
        # Show detailed breakdown
        with st.expander("📋 Detailed Breakdown"):
            st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Confidence Score Analysis")
        confidences = [d['confidence'] for d in st.session_state.detection_history]
        
        # Create confidence bins for histogram
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(confidences, bins=bins)
        
        # Create DataFrame for histogram
        hist_df = pd.DataFrame({
            'Confidence Range': [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)],
            'Frequency': hist
        })
        
        st.bar_chart(hist_df.set_index('Confidence Range')['Frequency'])
        
        # Show confidence statistics
        with st.expander("📊 Confidence Statistics"):
            st.metric("Min Confidence", f"{min(confidences):.3f}")
            st.metric("Max Confidence", f"{max(confidences):.3f}")
            st.metric("Median Confidence", f"{np.median(confidences):.3f}")
            st.metric("Std Deviation", f"{np.std(confidences):.3f}")
    
    # Detection timeline
    if len(st.session_state.detection_history) > 1:
        st.markdown("#### Detection Activity Timeline")
        df = pd.DataFrame(st.session_state.detection_history)
        df['hour'] = df['timestamp'].dt.hour
        hourly_counts = df.groupby('hour').size().reset_index(name='detections')
        
        # Fill missing hours with 0 detections
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly_counts = all_hours.merge(hourly_counts, on='hour', how='left').fillna(0)
        
        st.line_chart(hourly_counts.set_index('hour')['detections'])
        
        # Show peak activity time
        peak_hour = hourly_counts.loc[hourly_counts['detections'].idxmax(), 'hour']
        st.info(f"🕐 Peak activity time: {int(peak_hour)}:00 hours")
    
    # Recent detections table
    st.markdown("#### Recent Detections")
    recent_detections = st.session_state.detection_history[-10:]  # Last 10 detections
    
    if recent_detections:
        recent_df = pd.DataFrame([
            {
                'Time': d['timestamp'].strftime('%H:%M:%S'),
                'Object': d['class'],
                'Confidence': f"{d['confidence']:.2%}",
                'Bounding Box': f"({d['bbox'][0]}, {d['bbox'][1]}) - ({d['bbox'][2]}, {d['bbox'][3]})"
            }
            for d in recent_detections
        ])
        st.dataframe(recent_df, use_container_width=True)

class AdvancedVideoTransformer(VideoTransformerBase):
    """Enhanced video transformer with performance optimizations"""
    
    def __init__(self, settings):
        self.model = model
        self.settings = settings
        self.frame_count = 0
        self.skip_frames = 2  # Process every 3rd frame for better performance
        self.last_results = None
    
    def transform(self, frame):
        self.frame_count += 1
        
        # Skip frames for better performance
        if self.frame_count % self.skip_frames != 0 and self.last_results is not None:
            return self.last_results
        
        img_array = frame.to_ndarray(format="bgr24")
        
        # Make prediction
        results, _ = make_prediction(img_array, self.settings)
        
        if results is not None:
            img_with_bbox = create_advanced_bbox_image(img_array, results)
            self.last_results = img_with_bbox
            return img_with_bbox
        
        return img_array

def main():
    """Main application function"""
    # Apply styling
    apply_custom_css()
    
    # Create header
    create_header()
    
    # Create sidebar and get settings
    settings = create_sidebar()
    
    # Main content container
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image/Video Detection", "📹 Real-time Detection", "📊 Analytics", "⚙️ Advanced Settings"])
    
    with tab1:
        st.markdown("### Upload Media for Detection")
        
        upload = st.file_uploader(
            "Choose an image or video file",
            type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "webm"],
            help="Upload an image or video file for object detection analysis"
        )
        
        if upload is not None:
            if upload.type.startswith("image"):
                # Image processing
                img = Image.open(upload)
                img_array = np.array(img)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original Image")
                    st.image(img, use_column_width=True)
                
                with col2:
                    st.markdown("#### Detection Results")
                    with st.spinner("Analyzing image..."):
                        results, transform_params = make_prediction(img_array, settings)
                        
                        if results is not None:
                            img_with_bbox = create_advanced_bbox_image(img_array, results, transform_params)
                            st.image(img_with_bbox, use_column_width=True)
                            
                            # Show detection summary
                            detections = results.xyxyn[0]
                            if len(detections) > 0:
                                st.success(f"Found {len(detections)} objects!")
                                
                                # Create detection table
                                detection_data = []
                                for i, detection in enumerate(detections):
                                    class_id = int(detection[5])
                                    confidence = float(detection[4])
                                    class_name = model.names[class_id] if class_id < len(model.names) else f"Class_{class_id}"
                                    
                                    detection_data.append({
                                        'Object': class_name.replace('_', ' ').title(),
                                        'Confidence': f"{confidence:.2%}",
                                        'Box': f"({int(detection[0]*img_array.shape[1])}, {int(detection[1]*img_array.shape[0])}, {int(detection[2]*img_array.shape[1])}, {int(detection[3]*img_array.shape[0])})"
                                    })
                                
                                st.dataframe(pd.DataFrame(detection_data), use_container_width=True)
                            else:
                                st.warning("No objects detected. Try adjusting the confidence threshold.")
                        else:
                            st.error("Failed to process image. Please check your model configuration.")
            
            elif upload.type.startswith("video"):
                # Video processing
                st.markdown("#### Video Processing")
                st.video(upload)
                
                if st.button("🎬 Process Video", type="primary"):
                    # Save uploaded video temporarily
                    with open("temp_video.mp4", "wb") as f:
                        f.write(upload.read())
                    
                    cap = cv2.VideoCapture("temp_video.mp4")
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    st.info(f"Processing video: {total_frames} frames at {fps} FPS")
                    
                    # Create placeholders
                    progress_bar = st.progress(0)
                    frame_placeholder = st.empty()
                    stats_placeholder = st.empty()
                    
                    frame_count = 0
                    detection_counts = defaultdict(int)
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results, _ = make_prediction(frame_rgb, settings)
                        
                        if results is not None:
                            frame_with_bbox = create_advanced_bbox_image(frame_rgb, results)
                            frame_placeholder.image(frame_with_bbox, channels="RGB", use_column_width=True)
                            
                            # Update detection counts
                            for detection in results.xyxyn[0]:
                                if detection[4] >= settings['confidence']:
                                    class_id = int(detection[5])
                                    class_name = model.names[class_id] if class_id < len(model.names) else f"Class_{class_id}"
                                    detection_counts[class_name] += 1
                        
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        
                        # Update stats every 30 frames
                        if frame_count % 30 == 0:
                            stats_placeholder.json(dict(detection_counts))
                        
                        # Limit processing speed
                        time.sleep(0.01)
                    
                    cap.release()
                    os.remove("temp_video.mp4")
                    
                    st.success("Video processing completed!")
                    st.json(dict(detection_counts))
    
    with tab2:
        st.markdown("### Real-time Object Detection")
        st.markdown("Enable your camera to start real-time object detection.")
        
        # WebRTC configuration
        rtc_config = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        })
        
        # Real-time detection controls
        col1, col2 = st.columns(2)
        
        with col1:
            camera_enabled = st.checkbox("Enable Camera", value=False)
        
        with col2:
            detection_enabled = st.checkbox("Enable Detection", value=True)
        
        if camera_enabled:
            webrtc_streamer(
                key="realtime_detection",
                video_transformer_factory=lambda: AdvancedVideoTransformer(settings),
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
    
    with tab3:
        create_analytics_dashboard()
    
    with tab4:
        st.markdown("### Advanced Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Information")
            if model:
                st.success("✅ Model loaded successfully")
                st.info(f"Model type: {type(model).__name__}")
                st.info(f"Number of classes: {len(model.names)}")
                
                # Show class names
                with st.expander("View All Classes"):
                    for i, name in enumerate(model.names):
                        st.write(f"{i}: {name}")
            else:
                st.error("❌ Model not loaded")
        
        with col2:
            st.markdown("#### Performance Settings")
            
            # Clear cache button
            if st.button("🧹 Clear Cache", type="secondary"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared!")
            
            # Reset statistics
            if st.button("🔄 Reset Statistics", type="secondary"):
                st.session_state.detection_history = []
                st.session_state.detection_stats = defaultdict(int)
                st.success("Statistics reset!")
            
            # Export data
            if st.session_state.detection_history:
                df = pd.DataFrame(st.session_state.detection_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Detection Data",
                    data=csv,
                    file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
