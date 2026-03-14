import streamlit as st
import pandas as pd
import requests
import urllib.parse
import json
import re
import math
import time
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
from statistics import mean, stdev
import hashlib
import base64

# ================================================================
# FIREBASE IMPORTS (with graceful fallback)
# ================================================================

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# ================================================================
# OPTIMIZED IMPORTS - Handle BERT gracefully
# ================================================================

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="AI Fake Disaster Tweet Detector - Real-Time",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚨"
)

# ================================================================
# ENHANCED CSS WITH MODERN DESIGN
# ================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container with glass morphism */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        margin: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Header with gradient and animation */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        animation: slideDown 0.5s ease-out;
    }
    
    @keyframes slideDown {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Alert animations */
    .fake-alert, .real-alert, .hybrid-alert {
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        animation: pulse 2s infinite, slideIn 0.5s ease-out;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .fake-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
    }
    
    .real-alert {
        background: linear-gradient(135deg, #10ac84, #1dd1a1);
    }
    
    .hybrid-alert {
        background: linear-gradient(135deg, #f39c12, #e67e22);
    }
    
    /* Probability bar with gradient */
    .probability-bar-container {
        background: rgba(0,0,0,0.05);
        border-radius: 15px;
        padding: 5px;
        margin: 20px 0;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .probability-bar {
        height: 40px;
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .fake-bar {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #ee5253);
        color: white;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        transition: width 0.5s ease-in-out;
    }
    
    .real-bar {
        height: 100%;
        background: linear-gradient(90deg, #10ac84, #1dd1a1);
        color: white;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        transition: width 0.5s ease-in-out;
    }
    
    /* Metric cards with hover effects */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
        border-color: #667eea;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .metric-card h4 {
        color: #666;
        font-size: 0.9em;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h2 {
        font-size: 2em;
        margin: 0;
        font-weight: 700;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 0.9em;
        font-weight: 600;
        margin: 5px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-out;
    }
    
    .live-badge {
        background: linear-gradient(135deg, #ff6b6b, #ee5253);
        color: white;
    }
    
    .local-badge {
        background: linear-gradient(135deg, #95a5a6, #7f8c8d);
        color: white;
    }
    
    .model-badge {
        background: linear-gradient(135deg, #667eea, #5a67d8);
        color: white;
    }
    
    /* Input area with modern design */
    .stTextArea textarea {
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 15px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        background: white !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3) !important;
        transform: scale(1.02);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 12px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: none !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Primary button */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #667eea, #5a67d8) !important;
        color: white !important;
    }
    
    /* Example buttons */
    .example-button {
        background: white !important;
        color: #667eea !important;
        border: 2px solid #667eea !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-163ttbj {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
        margin: 20px;
    }
    
    /* Live feed table */
    .dataframe {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1) !important;
    }
    
    /* Alert container */
    .alert-container {
        background: linear-gradient(135deg, #fff5f5, #fff0f0);
        border-left: 5px solid #ff6b6b;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        animation: slideInRight 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Live indicator */
    .live-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #10ac84;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
    }
    
    .stat-value {
        font-size: 2.5em;
        font-weight: 700;
        color: #667eea;
        margin: 10px 0;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress bar */
    .progress-container {
        width: 100%;
        height: 10px;
        background: #f0f0f0;
        border-radius: 5px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #5a67d8);
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5em;
        font-weight: 700;
        color: #333;
        margin: 30px 0 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Grid layout */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 15px;
            margin: 10px;
        }
        
        .metric-card h2 {
            font-size: 1.5em;
        }
        
        .stat-value {
            font-size: 2em;
        }
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# FIREBASE INITIALIZATION (with graceful fallback)
# ================================================================

def initialize_firebase():
    """Initialize Firebase with fallback to local mode"""
    if not FIREBASE_AVAILABLE:
        return None, False
    
    if not firebase_admin._apps:
        try:
            # Try to get credentials from Streamlit secrets
            if "firebase" in st.secrets:
                firebase_config = {
                    "type": st.secrets["firebase"]["type"],
                    "project_id": st.secrets["firebase"]["project_id"],
                    "private_key_id": st.secrets["firebase"]["private_key_id"],
                    "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
                    "client_email": st.secrets["firebase"]["client_email"],
                    "client_id": st.secrets["firebase"]["client_id"],
                    "auth_uri": st.secrets["firebase"]["auth_uri"],
                    "token_uri": st.secrets["firebase"]["token_uri"]
                }
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                return db, True
            else:
                return None, False
        except Exception as e:
            return None, False
    else:
        return firestore.client(), True

# Initialize Firebase
db, FIREBASE_ACTIVE = initialize_firebase()

# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "hybrid"
if "input_key_counter" not in st.session_state:
    st.session_state["input_key_counter"] = 0
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "local_analyses" not in st.session_state:
    st.session_state["local_analyses"] = []
if "local_stats" not in st.session_state:
    st.session_state["local_stats"] = {
        "total_analyses": 0,
        "total_fake": 0,
        "total_real": 0,
        "locations": {},
        "disaster_types": {},
        "models_used": {}
    }
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "animations_enabled" not in st.session_state:
    st.session_state["animations_enabled"] = True

# ================================================================
# MAIN CONTAINER
# ================================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ================================================================
# HEADER WITH ANIMATION
# ================================================================
badge_class = "live-badge" if FIREBASE_ACTIVE else "local-badge"
badge_text = "🔴 LIVE" if FIREBASE_ACTIVE else "⚫ LOCAL"
connection_status = "Connected to Global Network" if FIREBASE_ACTIVE else "Offline Mode - Data stored locally"

st.markdown(
    f'''
    <div class="main-header">
        <h1 style="font-size: 3em; margin-bottom: 10px;">🚨 AI Fake Disaster Detector</h1>
        <p style="font-size: 1.2em; opacity: 0.9;">Real-time misinformation detection with advanced analytics</p>
        <div style="margin-top: 20px;">
            <span class="status-badge {badge_class}">{badge_text}</span>
            <span class="status-badge model-badge">🤖 {st.session_state["model_choice"].upper()}</span>
        </div>
        <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">{connection_status} | Session: {st.session_state["session_id"]}</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# ================================================================
# SIDEBAR WITH ENHANCED UI
# ================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # Connection Status Card
    st.markdown("### 🌐 Connection Status")
    if FIREBASE_ACTIVE:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10ac8420, #1dd1a120); padding: 15px; border-radius: 12px; border-left: 4px solid #10ac84;">
            <div style="display: flex; align-items: center;">
                <span class="live-dot"></span>
                <strong style="color: #10ac84;">LIVE CONNECTION</strong>
            </div>
            <p style="margin-top: 10px; font-size: 0.9em;">Data syncing globally in real-time</p>
            <div class="progress-container">
                <div class="progress-bar" style="width: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #95a5a620, #7f8c8d20); padding: 15px; border-radius: 12px; border-left: 4px solid #95a5a6;">
            <div style="display: flex; align-items: center;">
                <span style="width: 12px; height: 12px; background: #95a5a6; border-radius: 50%; margin-right: 8px;"></span>
                <strong style="color: #7f8c8d;">LOCAL MODE</strong>
            </div>
            <p style="margin-top: 10px; font-size: 0.9em;">Data stored locally - lost on refresh</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Status with Icons
    st.markdown("### 🤖 Model Status")
    col1, col2 = st.columns(2)
    with col1:
        if bert_loaded:
            st.markdown("""
            <div style="background: #10ac8420; padding: 10px; border-radius: 10px; text-align: center;">
                <span style="font-size: 2em;">🧠</span><br>
                <strong style="color: #10ac84;">BERT</strong><br>
                <span style="color: #10ac84;">✅ Active</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #ff6b6b20; padding: 10px; border-radius: 10px; text-align: center;">
                <span style="font-size: 2em;">🧠</span><br>
                <strong style="color: #ff6b6b;">BERT</strong><br>
                <span style="color: #ff6b6b;">❌ Unavailable</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #10ac8420; padding: 10px; border-radius: 10px; text-align: center;">
            <span style="font-size: 2em;">🔍</span><br>
            <strong style="color: #10ac84;">MOCK</strong><br>
            <span style="color: #10ac84;">✅ Always Ready</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings Card
    st.markdown("### ⚙️ Settings")
    
    # Model Selection with better UI
    if bert_loaded:
        model_choice = st.radio(
            "Select Detection Model",
            ["hybrid", "mock", "bert"],
            format_func=lambda x: {
                "hybrid": "🤖 Hybrid (BERT + Mock) - Most Accurate",
                "mock": "🔍 Mock Only - Fastest",
                "bert": "🧠 BERT Only - Deep Learning"
            }.get(x, x),
            index=0,
            help="Choose your preferred detection model"
        )
        st.session_state["model_choice"] = model_choice
    
    # Auto-refresh with toggle
    st.session_state["auto_refresh"] = st.toggle(
        "🔄 Auto-refresh Feed",
        value=True,
        help="Automatically refresh live feed every 5 seconds"
    )
    
    # Dark mode toggle (visual only)
    st.session_state["dark_mode"] = st.toggle(
        "🌙 Dark Mode",
        value=False,
        help="Toggle dark/light theme"
    )
    
    # Animations toggle
    st.session_state["animations_enabled"] = st.toggle(
        "✨ Enable Animations",
        value=True,
        help="Enable/disable UI animations"
    )
    
    st.markdown("---")
    
    # Live Statistics with Cards
    st.markdown("### 📊 Live Statistics")
    stats = rt_manager.get_live_stats() if 'rt_manager' in locals() else st.session_state["local_stats"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Analyses</div>
            <div class="stat-value">{stats.get('total_analyses', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Fake News</div>
            <div class="stat-value" style="color: #ff6b6b;">{stats.get('total_fake', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Real News</div>
            <div class="stat-value" style="color: #10ac84;">{stats.get('total_real', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if stats.get('locations'):
            top_location = max(stats['locations'].items(), key=lambda x: x[1])[0]
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Hotspot</div>
                <div class="stat-value" style="font-size: 1.2em;">📍 {top_location}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Clear data button
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state["local_analyses"] = []
        st.session_state["local_stats"] = {
            "total_analyses": 0,
            "total_fake": 0,
            "total_real": 0,
            "locations": {},
            "disaster_types": {},
            "models_used": {}
        }
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
# MAIN CONTENT AREA
# ================================================================

# Auto-refresh logic
if st.session_state["auto_refresh"]:
    if time.time() - st.session_state["last_refresh"] > 5:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

# ================================================================
# INPUT SECTION WITH ENHANCED UI
# ================================================================
st.markdown("### 📝 Tweet Analysis Input")

# Create a card-like input area
input_col1, input_col2 = st.columns([6, 1])

with input_col1:
    input_key = f"tweet_input_{st.session_state['input_key_counter']}"
    
    tweet = st.text_area(
        "Enter tweet to analyze:",
        height=120,
        placeholder="Paste or type a tweet here... Example: Heavy rain in Kampar causing flash floods - reported by local authorities",
        key=input_key,
        label_visibility="collapsed"
    )

with input_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear", use_container_width=True, help="Clear input field"):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# Quick examples with better styling
st.markdown("#### 🎯 Quick Examples")
example_col1, example_col2, example_col3, example_col4 = st.columns(4)

with example_col1:
    if st.button("📰 Real News", use_container_width=True, key="real_example"):
        st.session_state["tweet_input"] = "Heavy rain in Kampar causing flash floods. According to local authorities, JPS monitoring water levels."
        st.session_state["input_key_counter"] += 1
        st.rerun()

with example_col2:
    if st.button("🚨 Fake News", use_container_width=True, key="fake_example"):
        st.session_state["tweet_input"] = "URGENT! BREAKING: MASSIVE earthquake in Kuala Lumpur! Thousands DEAD! SHARE NOW! Government hiding truth! 😱😱😱"
        st.session_state["input_key_counter"] += 1
        st.rerun()

with example_col3:
    if st.button("🔄 Mixed", use_container_width=True, key="mixed_example"):
        st.session_state["tweet_input"] = "URGENT! Flood in Johor! Water level 2 meters! SHARE NOW! Official source says evacuating."
        st.session_state["input_key_counter"] += 1
        st.rerun()

with example_col4:
    if st.button("📍 Location Test", use_container_width=True, key="location_example"):
        st.session_state["tweet_input"] = "Flood in Penang - authorities confirm"
        st.session_state["input_key_counter"] += 1
        st.rerun()

# Action buttons
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze Tweet", type="primary", use_container_width=True)
with col2:
    if st.button("🔄 New Tweet", use_container_width=True):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# ================================================================
# REST OF YOUR EXISTING CODE (Analysis functions, display functions, etc.)
# ================================================================
# [Keep all your existing analysis functions here - analyze_mock, analyze_bert, analyze_hybrid]
# [Keep all your existing display functions - display_probability_bar, display_comprehensive_metrics, etc.]
# [Keep your RealtimeDataManager class]
# [Keep location functions]
# [Keep live feed display functions]

# ================================================================
# FOOTER WITH ENHANCED DESIGN
# ================================================================
st.markdown("---")
st.markdown(
    f'''
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea10, #764ba210); border-radius: 15px; margin-top: 30px;">
        <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 20px;">
            <span class="status-badge" style="background: #667eea; color: white;">🚀 v2.0</span>
            <span class="status-badge" style="background: #10ac84; color: white;">⚡ Real-time</span>
            <span class="status-badge" style="background: #f39c12; color: white;">🔬 AI-Powered</span>
        </div>
        <p style="color: #666; font-size: 0.9em;">
            AI Fake Disaster Tweet Detector | Real-time misinformation detection<br>
            Powered by BERT + Mock Analysis | Data {'syncing globally' if FIREBASE_ACTIVE else 'stored locally'}<br>
            Session: {st.session_state["session_id"]} | Last sync: {datetime.now().strftime('%H:%M:%S')}
        </p>
        <div style="margin-top: 20px;">
            <span class="live-dot"></span> Live system - Updates every 5 seconds
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
