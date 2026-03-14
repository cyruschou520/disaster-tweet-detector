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
    page_title="AI Fake Disaster Tweet Detector - GitHub Copilot API",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
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
        background: linear-gradient(135deg, #2b3137 0%, #1e2429 100%);
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
    
    /* Header with GitHub gradient */
    .main-header {
        background: linear-gradient(135deg, #2b3137 0%, #1e2429 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        animation: slideDown 0.5s ease-out;
    }
    
    @keyframes slideDown {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* GitHub Copilot badge */
    .copilot-badge {
        background: linear-gradient(135deg, #6e40ff, #a371f7);
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 1em;
        font-weight: 600;
        display: inline-block;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(110, 64, 255, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.9; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Alert animations */
    .fake-alert, .real-alert, .hybrid-alert, .copilot-alert {
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
    
    .copilot-alert {
        background: linear-gradient(135deg, #6e40ff, #a371f7);
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
        border: 1px solid rgba(110, 64, 255, 0.2);
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(110, 64, 255, 0.3);
        border-color: #6e40ff;
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
    
    .copilot-model-badge {
        background: linear-gradient(135deg, #6e40ff, #a371f7);
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
        border-color: #6e40ff !important;
        box-shadow: 0 5px 20px rgba(110, 64, 255, 0.3) !important;
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
        box-shadow: 0 10px 25px rgba(110, 64, 255, 0.4) !important;
    }
    
    /* Primary button */
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #6e40ff, #a371f7) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-163ttbj {
        background: linear-gradient(135deg, #2b3137 0%, #1e2429 100%) !important;
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
        box-shadow: 0 15px 30px rgba(110, 64, 255, 0.3);
    }
    
    .stat-value {
        font-size: 2.5em;
        font-weight: 700;
        color: #6e40ff;
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
        background: linear-gradient(90deg, #6e40ff, #a371f7);
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5em;
        font-weight: 700;
        color: #333;
        margin: 30px 0 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #6e40ff;
        display: inline-block;
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
    st.session_state["model_choice"] = "copilot"  # Default to Copilot
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
if "github_token" not in st.session_state:
    st.session_state["github_token"] = ""

# ================================================================
# CONSTANTS
# ================================================================
MALAYSIA_LOCATIONS = [
    'Kampar', 'Ipoh', 'Kuala Lumpur', 'KL', 'Penang', 'Pulau Pinang',
    'Johor', 'Johor Bahru', 'Shah Alam', 'Selangor', 'Perak', 'Kedah',
    'Kelantan', 'Terengganu', 'Pahang', 'Negeri Sembilan', 'Melaka',
    'Sabah', 'Sarawak', 'Langkawi', 'Kuantan', 'Kota Bharu', 'Alor Setar',
    'George Town', 'Butterworth', 'Taiping', 'Petaling Jaya', 'Subang Jaya',
    'Klang', 'Putrajaya', 'Cameron Highlands', 'Kota Kinabalu', 'Kuching'
]

DISASTER_KEYWORDS = {
    "flood": ['flood', 'banjir', 'inundation', 'water level'],
    "earthquake": ['earthquake', 'gempa', 'tremor', 'seismic'],
    "storm": ['storm', 'thunderstorm', 'ribut', 'petir', 'lightning'],
    "landslide": ['landslide', 'tanah runtuh', 'mudslide'],
    "fire": ['fire', 'kebakaran', 'burning'],
    "tsunami": ['tsunami', 'tidal wave'],
    "wind": ['wind', 'angin', 'gale', 'tornado']
}

FAKE_PATTERNS = {
    "urgency": ['urgent', 'breaking', 'alert', 'warning', '!!!'],
    "sensational": ['unbelievable', 'shocking', 'massive', 'worst ever', 'catastrophic'],
    "vague": ['they say', 'rumors', 'allegedly', 'someone said', 'apparently'],
    "sharing": ['share', 'viral', 'spread', 'forward', 'retweet'],
    "conspiracy": ['government hiding', 'they don\'t want you', 'secret', 'hidden truth']
}

REAL_PATTERNS = {
    "sources": ['according to', 'reported by', 'official', 'authorities', 'confirmed'],
    "specific": ['at', 'on', 'date', 'time', 'location', 'coordinates'],
    "measured": ['meter', 'km', 'mm', 'celsius', 'magnitude', 'level'],
    "organizations": ['JPS', 'MET Malaysia', 'Jabatan Bomba', 'police', 'nadma']
}

# ================================================================
# GITHUB COPILOT API FUNCTIONS
# ================================================================

def analyze_with_copilot(text, github_token):
    """
    Analyze tweet using GitHub Copilot API
    Note: This uses the same underlying model as GitHub Copilot
    """
    if not github_token:
        st.error("⚠️ Please enter your GitHub token in the sidebar")
        return None
    
    start_time = time.time()
    
    # GitHub Copilot API endpoint (Note: This is a conceptual endpoint - actual implementation may vary)
    # In practice, you would need to use GitHub's API with appropriate authentication
    url = "https://api.github.com/copilot/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Enhanced prompt for fake news detection
    prompt = f"""You are an expert at detecting fake news and misinformation in disaster-related tweets.
    
Analyze this tweet and provide a detailed JSON response with fake news probability.

Tweet: "{text}"

Consider these factors:
1. Urgency markers (urgent, breaking, alert)
2. Sensationalism (unbelievable, shocking, massive)
3. Source credibility (official sources, authorities)
4. Specific details (time, location, numbers)
5. Sharing requests (share, viral, spread)
6. Conspiracy elements (government hiding, secret)
7. Emotional language (pray, help, omg)

Return a JSON object with:
{{
    "is_fake": boolean (true if likely fake/misinformation),
    "fake_probability": float (0-1 probability it's fake),
    "confidence": float (0-1 confidence in this assessment),
    "reasons": [list of specific reasons],
    "urgency_level": "low/medium/high/critical",
    "sensationalism_score": float (0-1),
    "credibility_score": float (0-1),
    "detected_fake_indicators": {{
        "urgency": [list of urgency words found],
        "sensational": [list of sensational words found],
        "vague": [list of vague phrases found],
        "sharing": [list of sharing requests found],
        "conspiracy": [list of conspiracy indicators found]
    }},
    "detected_real_indicators": {{
        "sources": [list of official sources mentioned],
        "specific": [list of specific details found],
        "measured": [list of measurements found],
        "organizations": [list of official organizations mentioned]
    }}
}}

Return ONLY valid JSON, no other text."""
    
    payload = {
        "model": "gpt-4",  # Copilot uses GPT-4 based models
        "messages": [
            {"role": "system", "content": "You are an expert fake news detector. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Extract the JSON from the response
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Find JSON in the response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            
            # Add metadata
            analysis["response_time"] = time.time() - start_time
            analysis["model_used"] = "GitHub Copilot (GPT-4)"
            
            # Ensure probabilities sum to 1
            if "fake_probability" in analysis:
                analysis["real_probability"] = 1 - analysis["fake_probability"]
            else:
                analysis["fake_probability"] = 0.5
                analysis["real_probability"] = 0.5
            
            # Add text metrics
            words = text.split()
            analysis["word_count"] = len(words)
            analysis["exclamation_count"] = text.count('!')
            analysis["question_count"] = text.count('?')
            analysis["caps_words"] = sum(1 for word in words if word.isupper() and len(word) > 2)
            
            return analysis
        else:
            st.error("Could not parse Copilot response")
            return None
            
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            st.error("❌ Invalid GitHub token. Please check your token.")
        elif response.status_code == 429:
            st.error("⚠️ Rate limit exceeded. Please try again later.")
        elif response.status_code == 403:
            st.error("❌ Access forbidden. Check if Copilot API is enabled.")
        else:
            st.error(f"Copilot API error: {e}")
        return None
    except Exception as e:
        st.error(f"Error calling Copilot API: {e}")
        return None

# ================================================================
# FALLBACK MOCK ANALYSIS (when Copilot is unavailable)
# ================================================================

def analyze_mock_fallback(text):
    """Fallback mock analysis when Copilot is unavailable"""
    start_time = time.time()
    text_lower = text.lower()
    
    fake_score = 0
    real_score = 0
    fake_details = {category: 0 for category in FAKE_PATTERNS}
    real_details = {category: 0 for category in REAL_PATTERNS}
    
    # Check fake patterns
    for category, patterns in FAKE_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                fake_score += 1
                fake_details[category] += 1
    
    # Check real patterns
    for category, patterns in REAL_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                real_score += 1.5
                real_details[category] += 1
    
    # Detect disasters
    detected_disasters = []
    for disaster, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_disasters.append(disaster)
    
    # Word analysis
    words = text.split()
    word_count = len(words)
    
    # Calculate probabilities
    total = fake_score + real_score
    if total > 0:
        fake_prob = fake_score / total
    else:
        fake_prob = 0.5
    
    real_prob = 1 - fake_prob
    
    return {
        "is_fake": fake_prob > 0.5,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "confidence": abs(fake_prob - 0.5) * 2,
        "reasons": ["Using fallback mock analysis - Copilot unavailable"],
        "urgency_level": "medium",
        "sensationalism_score": fake_details.get("sensational", 0) / 5,
        "credibility_score": real_details.get("sources", 0) / 4,
        "detected_disasters": detected_disasters,
        "word_count": word_count,
        "exclamation_count": text.count('!'),
        "question_count": text.count('?'),
        "caps_words": sum(1 for word in words if word.isupper() and len(word) > 2),
        "model_used": "Mock Fallback",
        "response_time": time.time() - start_time
    }

# ================================================================
# BERT ANALYSIS (keep as backup)
# ================================================================

@st.cache_resource(show_spinner="Loading BERT model...")
def load_bert_model():
    """Load BERT model with graceful fallback"""
    if not BERT_AVAILABLE:
        return None, None, False
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer, True
    except:
        return None, None, False

bert_model, bert_tokenizer, bert_loaded = load_bert_model() if BERT_AVAILABLE else (None, None, False)

def analyze_bert(text):
    """BERT analysis as backup"""
    if not bert_loaded or bert_model is None:
        return None
    
    start_time = time.time()
    
    try:
        inputs = bert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)[0]
        fake_prob = probs[1].item()
        real_prob = probs[0].item()
        
        total = fake_prob + real_prob
        fake_prob = fake_prob / total
        real_prob = real_prob / total
        
        return {
            "is_fake": fake_prob > 0.5,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": abs(fake_prob - 0.5) * 2,
            "reasons": ["BERT model analysis"],
            "model_used": "BERT",
            "response_time": time.time() - start_time,
            "word_count": len(text.split()),
            "exclamation_count": text.count('!'),
            "question_count": text.count('?')
        }
    except Exception as e:
        return None

# ================================================================
# MAIN ANALYSIS FUNCTION
# ================================================================

def analyze_tweet(text, model_choice, github_token):
    """Main analysis function routing"""
    
    if model_choice == "copilot":
        result = analyze_with_copilot(text, github_token)
        if result is None:
            st.warning("⚠️ Copilot API failed - falling back to mock analysis")
            result = analyze_mock_fallback(text)
    elif model_choice == "bert" and bert_loaded:
        result = analyze_bert(text)
        if result is None:
            result = analyze_mock_fallback(text)
    else:
        result = analyze_mock_fallback(text)
    
    return result

# ================================================================
# REAL-TIME DATA MANAGER
# ================================================================

class RealtimeDataManager:
    """Manages real-time data synchronization with Firebase or local storage"""
    
    def __init__(self, db, firebase_active):
        self.db = db
        self.firebase_active = firebase_active
        self.analyses_collection = "tweet_analyses"
        self.stats_collection = "system_stats"
        self.alerts_collection = "active_alerts"
        
    def save_analysis(self, analysis_data):
        """Save analysis to Firebase if available, otherwise store locally"""
        if not self.firebase_active:
            st.session_state["local_analyses"].append(analysis_data)
            stats = st.session_state["local_stats"]
            stats["total_analyses"] += 1
            if analysis_data.get("is_fake"):
                stats["total_fake"] += 1
            else:
                stats["total_real"] += 1
            
            if analysis_data.get("location"):
                loc = analysis_data["location"]
                stats["locations"][loc] = stats["locations"].get(loc, 0) + 1
            
            for disaster in analysis_data.get("detected_disasters", []):
                stats["disaster_types"][disaster] = stats["disaster_types"].get(disaster, 0) + 1
            
            model = analysis_data.get("model_used", "unknown")
            stats["models_used"][model] = stats["models_used"].get(model, 0) + 1
            
            return "local"
        
        try:
            doc_ref = self.db.collection(self.analyses_collection).document()
            analysis_data["timestamp"] = firestore.SERVER_TIMESTAMP
            analysis_data["id"] = doc_ref.id
            doc_ref.set(analysis_data)
            
            self.update_stats(analysis_data)
            
            if analysis_data.get("is_fake") and analysis_data.get("confidence", 0) > 0.8:
                self.create_alert(analysis_data)
                
            return doc_ref.id
        except Exception as e:
            st.error(f"Firebase error: {e}")
            return None
    
    def update_stats(self, analysis_data):
        """Update real-time statistics"""
        try:
            stats_ref = self.db.collection(self.stats_collection).document("live_stats")
            
            @firestore.transactional
            def update_in_transaction(transaction, stats_ref):
                snapshot = stats_ref.get(transaction=transaction)
                if snapshot.exists:
                    stats = snapshot.to_dict()
                else:
                    stats = {
                        "total_analyses": 0,
                        "total_fake": 0,
                        "total_real": 0,
                        "locations": {},
                        "disaster_types": {},
                        "models_used": {},
                        "last_24h": []
                    }
                
                stats["total_analyses"] += 1
                if analysis_data.get("is_fake"):
                    stats["total_fake"] += 1
                else:
                    stats["total_real"] += 1
                
                if analysis_data.get("location"):
                    loc = analysis_data["location"]
                    stats["locations"][loc] = stats["locations"].get(loc, 0) + 1
                
                for disaster in analysis_data.get("detected_disasters", []):
                    stats["disaster_types"][disaster] = stats["disaster_types"].get(disaster, 0) + 1
                
                model = analysis_data.get("model_used", "unknown")
                stats["models_used"][model] = stats["models_used"].get(model, 0) + 1
                
                stats["last_24h"].append({
                    "timestamp": datetime.now().isoformat(),
                    "is_fake": analysis_data.get("is_fake")
                })
                if len(stats["last_24h"]) > 1000:
                    stats["last_24h"] = stats["last_24h"][-1000:]
                
                transaction.set(stats_ref, stats)
                return stats
            
            transaction = self.db.transaction()
            update_in_transaction(transaction, stats_ref)
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def create_alert(self, analysis_data):
        """Create real-time alert for high-confidence fake news"""
        try:
            alert_ref = self.db.collection(self.alerts_collection).document()
            alert_data = {
                "tweet": analysis_data.get("tweet", "")[:100],
                "location": analysis_data.get("location", "Unknown"),
                "confidence": analysis_data.get("confidence", 0),
                "timestamp": firestore.SERVER_TIMESTAMP,
                "status": "active",
                "id": alert_ref.id
            }
            alert_ref.set(alert_data)
        except Exception as e:
            print(f"Error creating alert: {e}")
    
    def get_live_analyses(self, limit=50):
        """Get latest analyses from Firebase or local storage"""
        if not self.firebase_active:
            return st.session_state["local_analyses"][-limit:]
        
        try:
            analyses = self.db.collection(self.analyses_collection)\
                .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .stream()
            return [doc.to_dict() for doc in analyses]
        except Exception as e:
            return st.session_state["local_analyses"][-limit:]
    
    def get_live_stats(self):
        """Get real-time statistics"""
        if not self.firebase_active:
            return st.session_state["local_stats"]
        
        try:
            stats_ref = self.db.collection(self.stats_collection).document("live_stats")
            stats = stats_ref.get()
            return stats.to_dict() if stats.exists else {}
        except Exception as e:
            return st.session_state["local_stats"]
    
    def get_active_alerts(self):
        """Get active fake news alerts"""
        if not self.firebase_active:
            return []
        
        try:
            alerts = self.db.collection(self.alerts_collection)\
                .where("status", "==", "active")\
                .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                .limit(10)\
                .stream()
            return [doc.to_dict() for doc in alerts]
        except Exception as e:
            return []
    
    def resolve_alert(self, alert_id):
        """Mark alert as resolved"""
        if not self.firebase_active:
            return
        
        try:
            self.db.collection(self.alerts_collection).document(alert_id)\
                .update({"status": "resolved", "resolved_at": firestore.SERVER_TIMESTAMP})
        except Exception as e:
            print(f"Error resolving alert: {e}")

# Initialize real-time manager
rt_manager = RealtimeDataManager(db, FIREBASE_ACTIVE)

# ================================================================
# DISPLAY FUNCTIONS
# ================================================================

def display_probability_bar(fake_prob, real_prob):
    """Display visual probability bar"""
    fake_percent = fake_prob * 100
    real_percent = real_prob * 100
    
    st.markdown(f"""
    <div class="probability-bar-container">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <span style="color: #ff6b6b; font-weight: 600;">FAKE: {fake_percent:.1f}%</span>
            <span style="color: #10ac84; font-weight: 600;">REAL: {real_percent:.1f}%</span>
        </div>
        <div class="probability-bar">
            <div class="fake-bar" style="width: {fake_percent}%;">
                {fake_percent:.1f}%
            </div>
            <div class="real-bar" style="width: {real_percent}%;">
                {real_percent:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_comprehensive_metrics(analysis):
    """Display all analysis metrics"""
    
    st.markdown('<h3 class="section-header">📊 Analysis Metrics</h3>', unsafe_allow_html=True)
    
    # Probability Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Fake Probability</h4>
            <h2 style="color: {'#ff6b6b' if analysis['fake_probability'] > 0.5 else '#666'}">
                {analysis['fake_probability']*100:.1f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Real Probability</h4>
            <h2 style="color: {'#10ac84' if analysis['real_probability'] > 0.5 else '#666'}">
                {analysis['real_probability']*100:.1f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Decision</h4>
            <h2>{'❌ FAKE' if analysis['is_fake'] else '✅ REAL'}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence Analysis
    st.markdown('<h4 style="margin-top: 20px;">🎯 Confidence Analysis</h4>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Confidence", f"{analysis.get('confidence', 0.5)*100:.1f}%")
    
    with col2:
        st.metric("Urgency", analysis.get('urgency_level', 'medium').upper())
    
    with col3:
        st.metric("Sensationalism", f"{analysis.get('sensationalism_score', 0)*100:.1f}%")
    
    with col4:
        st.metric("Credibility", f"{analysis.get('credibility_score', 0)*100:.1f}%")
    
    # Text Analysis
    st.markdown('<h4 style="margin-top: 20px;">📝 Text Analysis</h4>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Words", analysis.get('word_count', 0))
    with col2:
        st.metric("Exclamation", analysis.get('exclamation_count', 0))
    with col3:
        st.metric("Questions", analysis.get('question_count', 0))
    with col4:
        st.metric("Caps Words", analysis.get('caps_words', 0))
    
    # Reasons
    if analysis.get('reasons'):
        st.markdown("#### 🔍 Detection Reasons")
        for reason in analysis['reasons']:
            st.warning(f"• {reason}")
    
    # Disaster Detection
    if analysis.get('detected_disasters'):
        st.markdown("#### 🌪️ Detected Disaster Types")
        cols = st.columns(len(analysis['detected_disasters']))
        for i, disaster in enumerate(analysis['detected_disasters']):
            with cols[i]:
                st.info(disaster.upper())

def create_location_map(location, lat, lon, is_fake):
    """Create a location map"""
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        marker=dict(
            size=20,
            color='red' if is_fake else 'green',
            symbol='marker'
        ),
        text=[location],
        textposition="top center"
    ))
    
    radius_km = 5
    radius_deg = radius_km / 111.0
    circle_points = 50
    circle_lats = [lat + radius_deg * math.cos(2 * math.pi * i / circle_points) 
                  for i in range(circle_points + 1)]
    circle_lons = [lon + radius_deg * math.sin(2 * math.pi * i / circle_points) 
                  for i in range(circle_points + 1)]
    
    fig.add_trace(go.Scattermapbox(
        lat=circle_lats,
        lon=circle_lons,
        mode='lines',
        line=dict(width=2, color='red' if is_fake else 'green'),
        name='5km Radius'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lon),
            zoom=10
        ),
        height=400,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    return fig

def display_live_stats():
    """Display real-time statistics"""
    
    stats = rt_manager.get_live_stats()
    
    if not stats:
        st.info("Waiting for first analysis...")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", stats.get("total_analyses", 0))
    with col2:
        st.metric("Fake News", stats.get("total_fake", 0))
    with col3:
        st.metric("Real News", stats.get("total_real", 0))
    with col4:
        if "last_24h" in stats and stats["last_24h"]:
            last_hour = sum(1 for x in stats["last_24h"] 
                          if (datetime.now() - datetime.fromisoformat(x["timestamp"])).total_seconds() < 3600)
            st.metric("Last Hour", last_hour)
    
    if stats.get("locations"):
        st.markdown("### 📍 Top Locations")
        loc_df = pd.DataFrame([
            {"Location": loc, "Count": count}
            for loc, count in stats["locations"].items()
        ]).sort_values("Count", ascending=False).head(10)
        
        fig = px.bar(loc_df, x="Location", y="Count", 
                     title="Most Active Locations",
                     color="Count", color_continuous_scale="purples")
        st.plotly_chart(fig, use_container_width=True)

def display_live_alerts():
    """Display real-time alerts"""
    
    alerts = rt_manager.get_active_alerts()
    
    if alerts:
        st.markdown("### 🚨 Active Alerts")
        
        for alert in alerts:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.error(f"⚠️ {alert.get('tweet', '')}")
                with col2:
                    st.warning(alert.get('location', 'Unknown'))
                with col3:
                    st.warning(f"{alert.get('confidence', 0)*100:.0f}%")
                with col4:
                    if st.button(f"Resolve", key=f"resolve_{alert['id']}"):
                        rt_manager.resolve_alert(alert['id'])
                        st.rerun()

def display_live_feed():
    """Display live feed of recent analyses"""
    
    analyses = rt_manager.get_live_analyses(limit=20)
    
    if analyses:
        st.markdown("### 📡 Live Analysis Feed")
        
        feed_data = []
        for a in analyses:
            timestamp = a.get("timestamp", "")
            if hasattr(timestamp, "strftime"):
                timestamp = timestamp.strftime("%H:%M:%S")
            elif isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
                except:
                    timestamp = timestamp[:8]
            
            feed_data.append({
                "Time": timestamp,
                "Tweet": a.get("tweet_preview", a.get("tweet", "")[:50] + "..."),
                "Fake %": f"{a.get('fake_probability', 0)*100:.1f}%",
                "Location": a.get("location", "Unknown"),
                "Status": "❌ FAKE" if a.get("is_fake") else "✅ REAL",
                "Model": a.get("model_used", "Unknown")
            })
        
        df = pd.DataFrame(feed_data)
        st.dataframe(df, use_container_width=True, height=300)

# ================================================================
# MAIN UI STARTS HERE
# ================================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
badge_class = "live-badge" if FIREBASE_ACTIVE else "local-badge"
badge_text = "🔴 LIVE" if FIREBASE_ACTIVE else "⚫ LOCAL"
connection_status = "Connected to Global Network" if FIREBASE_ACTIVE else "Offline Mode - Data stored locally"

st.markdown(
    f'''
    <div class="main-header">
        <h1 style="font-size: 3em; margin-bottom: 10px;">🤖 GitHub Copilot Detector</h1>
        <p style="font-size: 1.2em; opacity: 0.9;">AI-powered fake news detection with GitHub Copilot API</p>
        <div style="margin-top: 20px;">
            <span class="copilot-badge">⚡ Powered by GitHub Copilot</span>
            <span class="status-badge {badge_class}">{badge_text}</span>
        </div>
        <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">{connection_status} | Session: {st.session_state["session_id"]}</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# ================================================================
# SIDEBAR
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
    
    # GitHub Token Input
    st.markdown("### 🔑 GitHub Authentication")
    github_token = st.text_input(
        "GitHub Token",
        type="password",
        value=st.session_state.get("github_token", ""),
        help="Enter your GitHub token with Copilot API access"
    )
    st.session_state["github_token"] = github_token
    
    if github_token:
        st.success("✅ Token provided")
    else:
        st.warning("⚠️ Enter token for Copilot API")
    
    st.markdown("---")
    
    # Model Status
    st.markdown("### 🤖 Model Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #6e40ff20; padding: 10px; border-radius: 10px; text-align: center;">
            <span style="font-size: 2em;">🤖</span><br>
            <strong style="color: #6e40ff;">Copilot</strong><br>
            <span style="color: #6e40ff;">✅ Active</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if bert_loaded:
            st.markdown("""
            <div style="background: #10ac8420; padding: 10px; border-radius: 10px; text-align: center;">
                <span style="font-size: 2em;">🧠</span><br>
                <strong style="color: #10ac84;">BERT</strong><br>
                <span style="color: #10ac84;">✅ Ready</span>
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
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
    # Model Selection
    if bert_loaded:
        model_options = ["copilot", "bert"]
    else:
        model_options = ["copilot"]
    
    model_labels = {
        "copilot": "🤖 GitHub Copilot (GPT-4) - Recommended",
        "bert": "🧠 BERT Only - Local"
    }
    
    model_choice = st.radio(
        "Select Detection Model",
        model_options,
        format_func=lambda x: model_labels.get(x, x),
        index=0,
        help="Choose your preferred detection model"
    )
    st.session_state["model_choice"] = model_choice
    
    # Auto-refresh toggle
    st.session_state["auto_refresh"] = st.toggle(
        "🔄 Auto-refresh Feed",
        value=True,
        help="Automatically refresh live feed every 5 seconds"
    )
    
    # Animations toggle
    st.session_state["animations_enabled"] = st.toggle(
        "✨ Enable Animations",
        value=True,
        help="Enable/disable UI animations"
    )
    
    st.markdown("---")
    
    # Live Statistics
    st.markdown("### 📊 Live Statistics")
    stats = rt_manager.get_live_stats()
    
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
# AUTO-REFRESH LOGIC
# ================================================================
if st.session_state["auto_refresh"]:
    if time.time() - st.session_state["last_refresh"] > 5:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

# ================================================================
# LIVE ALERTS SECTION
# ================================================================
display_live_alerts()

# ================================================================
# INPUT SECTION
# ================================================================
st.markdown("### 📝 Tweet Analysis Input")

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

# Quick Examples
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

# Action Buttons
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze with Copilot", type="primary", use_container_width=True)
with col2:
    if st.button("🔄 New Tweet", use_container_width=True):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# ================================================================
# ANALYSIS EXECUTION
# ================================================================
if analyze_clicked and tweet:
    with st.spinner("🤖 Calling GitHub Copilot API..."):
        
        result = analyze_tweet(
            tweet, 
            st.session_state["model_choice"],
            st.session_state["github_token"]
        )
        
        if result:
            # Extract location
            location = None
            for loc in MALAYSIA_LOCATIONS:
                if loc.lower() in tweet.lower():
                    location = loc
                    break
            
            # Prepare data for storage
            analysis_data = {
                "tweet": tweet,
                "tweet_preview": tweet[:100] + "..." if len(tweet) > 100 else tweet,
                "location": location,
                "is_fake": result.get("is_fake"),
                "fake_probability": result.get("fake_probability"),
                "real_probability": result.get("real_probability"),
                "confidence": result.get("confidence", 0.5),
                "detected_disasters": result.get("detected_disasters", []),
                "word_count": result.get("word_count", 0),
                "model_used": result.get("model_used", "Copilot"),
                "session_id": st.session_state["session_id"]
            }
            
            # Save to Firebase or local storage
            doc_id = rt_manager.save_analysis(analysis_data)
            
            if doc_id and FIREBASE_ACTIVE:
                st.success(f"✅ Analysis saved to global feed!")
            else:
                st.success("✅ Analysis saved locally")
            
            # Display results
            st.markdown("---")
            
            # Alert based on result
            if result["is_fake"]:
                st.markdown(
                    f'<div class="copilot-alert">🤖 COPILOT DETECTED: FAKE NEWS<br>'
                    f'Confidence: {result.get("confidence", 0.5)*100:.1f}% | '
                    f'Fake: {result["fake_probability"]*100:.1f}%</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="real-alert">✅ REAL NEWS<br>'
                    f'Confidence: {result.get("confidence", 0.5)*100:.1f}% | '
                    f'Real: {result["real_probability"]*100:.1f}%</div>',
                    unsafe_allow_html=True
                )
            
            # Display probability bar
            display_probability_bar(result["fake_probability"], result["real_probability"])
            
            # Display comprehensive metrics
            display_comprehensive_metrics(result)
            
            # Display location map if location found
            if location:
                st.info(f"📍 Location detected: {location}")
                
                try:
                    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location)}&format=json&limit=1"
                    headers = {'User-Agent': 'Disaster-Detector/1.0'}
                    response = requests.get(url, headers=headers, timeout=5)
                    data = response.json()
                    
                    if data:
                        lat, lon = float(data[0]['lat']), float(data[0]['lon'])
                        st.markdown("### 🗺️ Location Map")
                        fig = create_location_map(location, lat, lon, result["is_fake"])
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not load map for {location}")
            
            # Model info
            st.info(f"🤖 Model: {result.get('model_used', 'Copilot')} | Response time: {result.get('response_time', 0)*1000:.0f}ms")

elif analyze_clicked and not tweet:
    st.warning("⚠️ Please enter a tweet to analyze.")

# ================================================================
# LIVE FEED
# ================================================================
st.markdown("---")
display_live_feed()

if st.button("🔄 Refresh Live Feed", use_container_width=True):
    st.rerun()

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    f'''
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #6e40ff10, #a371f710); border-radius: 15px; margin-top: 30px;">
        <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 20px;">
            <span class="status-badge" style="background: #6e40ff; color: white;">🤖 GitHub Copilot</span>
            <span class="status-badge" style="background: #10ac84; color: white;">⚡ Real-time</span>
            <span class="status-badge" style="background: #f39c12; color: white;">🔬 AI-Powered</span>
        </div>
        <p style="color: #666; font-size: 0.9em;">
            AI Fake Disaster Tweet Detector | Powered by GitHub Copilot API<br>
            Using GPT-4 for advanced fake news detection | Data {'syncing globally' if FIREBASE_ACTIVE else 'stored locally'}<br>
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
