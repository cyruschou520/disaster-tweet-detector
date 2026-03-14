import streamlit as st
import pandas as pd
import requests
import urllib.parse
import json
import re
import math
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ================================================================
# OPTIMIZED IMPORTS - Handle BERT gracefully
# ================================================================

# Try to import BERT, but don't fail if not available
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    st.sidebar.info("📦 BERT not installed - using Mock Mode only")

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="AI Fake Disaster Tweet Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 20px; border-radius: 10px;
    color: white; text-align: center; margin-bottom: 30px;
}
.fake-alert {
    background: linear-gradient(135deg, #ff4444, #cc0000);
    color: white; padding: 15px 20px; border-radius: 10px;
    font-size: 1.1em; font-weight: bold;
}
.real-alert {
    background: linear-gradient(135deg, #00cc66, #008844);
    color: white; padding: 15px 20px; border-radius: 10px;
    font-size: 1.1em; font-weight: bold;
}
.hybrid-alert {
    background: linear-gradient(135deg, #ffaa00, #ff8800);
    color: white; padding: 15px 20px; border-radius: 10px;
    font-size: 1.1em; font-weight: bold;
}
.probability-bar {
    width: 100%;
    height: 30px;
    background-color: #f0f0f0;
    border-radius: 15px;
    margin: 10px 0;
    overflow: hidden;
    display: flex;
}
.fake-bar {
    height: 100%;
    background: linear-gradient(90deg, #ff4444, #cc0000);
    color: white;
    text-align: center;
    line-height: 30px;
    font-weight: bold;
}
.real-bar {
    height: 100%;
    background: linear-gradient(90deg, #00cc66, #008844);
    color: white;
    text-align: center;
    line-height: 30px;
    font-weight: bold;
}
.model-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: bold;
    margin-left: 10px;
}
.bert-badge {
    background: #667eea;
    color: white;
}
.mock-badge {
    background: #48bb78;
    color: white;
}
.hybrid-badge {
    background: #ffaa00;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">'
    "<h1>🚨 AI Fake Disaster Tweet Detector</h1>"
    "<p>Hybrid detection system - BERT + Mock Analysis</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ================================================================
# SESSION STATE
# ================================================================
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []
if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = ""
if "analysis_cache" not in st.session_state:
    st.session_state["analysis_cache"] = {}
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "hybrid"

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

# Fake news patterns
FAKE_PATTERNS = {
    "urgency": ['urgent', 'breaking', 'alert', 'warning', '!!!'],
    "sensational": ['unbelievable', 'shocking', 'massive', 'worst ever'],
    "vague": ['they say', 'rumors', 'allegedly', 'someone said'],
    "sharing": ['share', 'viral', 'spread', 'forward']
}

REAL_PATTERNS = {
    "sources": ['according to', 'reported by', 'official', 'authorities'],
    "specific": ['at', 'on', 'date', 'time', 'location'],
    "measured": ['meter', 'km', 'mm', 'celsius', 'magnitude']
}

# ================================================================
# BERT MODEL LOADING (with fallback)
# ================================================================

@st.cache_resource(show_spinner="Loading BERT model...")
def load_bert_model():
    """Load BERT model with graceful fallback"""
    if not BERT_AVAILABLE:
        return None, None, False
    
    try:
        # Try to load a lightweight model first
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer, True
    except:
        try:
            # Fallback to even smaller model
            model_name = "google/bert_uncased_L-2_H-128_A-2"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer, True
        except Exception as e:
            st.sidebar.warning(f"⚠️ BERT not available: {str(e)[:50]}...")
            return None, None, False

# Load model (will return None if not available)
bert_model, bert_tokenizer, bert_loaded = load_bert_model() if BERT_AVAILABLE else (None, None, False)

# ================================================================
# MOCK ANALYSIS (Always Available)
# ================================================================

def analyze_mock(text):
    """Deterministic mock analysis"""
    text_lower = text.lower()
    
    fake_score = 0
    real_score = 0
    
    # Check fake patterns
    for category, patterns in FAKE_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                fake_score += 1
    
    # Check real patterns
    for category, patterns in REAL_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                real_score += 1.5
    
    # Word count
    words = text.split()
    if len(words) > 30:
        real_score += 2
    elif len(words) < 5:
        fake_score += 1
    
    # Exclamation marks
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        fake_score += exclamation_count * 0.3
    
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
        "fake_score": fake_score,
        "real_score": real_score,
        "exclamation_count": exclamation_count,
        "word_count": len(words),
        "model": "Mock"
    }

# ================================================================
# BERT ANALYSIS (if available)
# ================================================================

def analyze_bert(text):
    """BERT analysis - returns None if not available"""
    if not bert_loaded or bert_model is None:
        return None
    
    try:
        inputs = bert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128  # Shorter for speed
        )
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)[0]
        fake_prob = probs[1].item()
        real_prob = probs[0].item()
        
        # Normalize
        total = fake_prob + real_prob
        fake_prob = fake_prob / total
        real_prob = real_prob / total
        
        return {
            "is_fake": fake_prob > 0.5,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": abs(fake_prob - 0.5) * 2,
            "model": "BERT"
        }
    except Exception as e:
        st.error(f"BERT analysis error: {e}")
        return None

# ================================================================
# HYBRID ANALYSIS
# ================================================================

def analyze_hybrid(text):
    """Combine BERT and Mock analysis"""
    mock_result = analyze_mock(text)
    bert_result = analyze_bert(text) if bert_loaded else None
    
    if bert_result:
        # Weighted average: BERT 60%, Mock 40%
        fake_prob = (bert_result["fake_probability"] * 0.6 + 
                    mock_result["fake_probability"] * 0.4)
        
        return {
            "is_fake": fake_prob > 0.5,
            "fake_probability": fake_prob,
            "real_probability": 1 - fake_prob,
            "confidence": abs(fake_prob - 0.5) * 2,
            "bert_contribution": bert_result["fake_probability"],
            "mock_contribution": mock_result["fake_probability"],
            "mock_details": mock_result,
            "model": "Hybrid (BERT+Mock)" if bert_loaded else "Mock Only",
            "bert_available": bert_loaded
        }
    else:
        # Fallback to mock only
        result = mock_result
        result["model"] = "Mock Only (BERT unavailable)"
        return result

# ================================================================
# LOCATION FUNCTIONS
# ================================================================

def extract_location(text):
    """Extract location from text"""
    text_lower = text.lower()
    for loc in MALAYSIA_LOCATIONS:
        if loc.lower() in text_lower:
            return loc
    return None

def get_coordinates(location):
    """Get coordinates with error handling"""
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location)}&format=json&limit=1"
        headers = {'User-Agent': 'Disaster-Detector/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    except:
        pass
    return None, None

def create_map(location, lat, lon, is_fake):
    """Create a simple map"""
    fig = px.scatter_mapbox(
        lat=[lat], 
        lon=[lon],
        hover_name=[location],
        zoom=9,
        height=300
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    return fig

# ================================================================
# UI COMPONENTS
# ================================================================

def display_probability_bar(fake_prob, real_prob):
    """Display probability bar"""
    fake_percent = fake_prob * 100
    real_percent = real_prob * 100
    
    st.markdown(f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #ff4444;">FAKE: {fake_percent:.1f}%</span>
            <span style="color: #00cc66;">REAL: {real_percent:.1f}%</span>
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

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model status
    st.subheader("📊 Model Status")
    col1, col2 = st.columns(2)
    with col1:
        if bert_loaded:
            st.success("✅ BERT")
        else:
            st.error("❌ BERT")
    with col2:
        st.success("✅ Mock")
    
    # Model selection
    if bert_loaded:
        model_choice = st.radio(
            "Select Model",
            ["hybrid", "mock", "bert"],
            format_func=lambda x: {
                "hybrid": "🤖 Hybrid (Recommended)",
                "mock": "🔍 Mock Only",
                "bert": "🧠 BERT Only"
            }.get(x, x),
            index=0
        )
        st.session_state["model_choice"] = model_choice
    else:
        st.info("🔍 Using Mock Mode (BERT unavailable)")
        st.session_state["model_choice"] = "mock"
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📈 Statistics")
    st.metric("Total Analyses", len(st.session_state["analysis_history"]))
    if st.button("🗑️ Clear History"):
        st.session_state["analysis_history"] = []
        st.rerun()

# ================================================================
# MAIN UI
# ================================================================

# Show active model
model_display = {
    "hybrid": "🤖 Hybrid (BERT + Mock)",
    "bert": "🧠 BERT Only",
    "mock": "🔍 Mock Only"
}.get(st.session_state["model_choice"], "Unknown")

badge_class = {
    "hybrid": "hybrid-badge",
    "bert": "bert-badge",
    "mock": "mock-badge"
}.get(st.session_state["model_choice"], "")

st.markdown(f"""
<div style="margin-bottom: 20px;">
    <span class="model-badge {badge_class}">Active: {model_display}</span>
</div>
""", unsafe_allow_html=True)

# Input
tweet = st.text_area(
    "📝 Enter tweet to analyze:",
    height=100,
    placeholder="Example: Heavy rain in Kampar causing flash floods - reported by local authorities..."
)

# Analyze button
if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if tweet:
        with st.spinner("Analyzing..."):
            # Choose analysis method
            if st.session_state["model_choice"] == "bert" and bert_loaded:
                result = analyze_bert(tweet)
            elif st.session_state["model_choice"] == "mock":
                result = analyze_mock(tweet)
            else:  # hybrid
                result = analyze_hybrid(tweet)
            
            if result:
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Alert
                if result["is_fake"]:
                    st.markdown(
                        f'<div class="fake-alert">❌ FAKE NEWS DETECTED<br>'
                        f'Confidence: {result["confidence"]*100:.1f}%</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="real-alert">✅ REAL NEWS<br>'
                        f'Confidence: {result["confidence"]*100:.1f}%</div>',
                        unsafe_allow_html=True
                    )
                
                # Probability bar
                display_probability_bar(
                    result["fake_probability"],
                    result["real_probability"]
                )
                
                # Model info
                st.info(f"🤖 Model: {result.get('model', 'Unknown')}")
                
                # Show contributions for hybrid
                if 'bert_contribution' in result:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("BERT Contribution", f"{result['bert_contribution']*100:.1f}%")
                    with col2:
                        st.metric("Mock Contribution", f"{result['mock_contribution']*100:.1f}%")
                
                # Location
                location = extract_location(tweet)
                if location:
                    lat, lon = get_coordinates(location)
                    if lat and lon:
                        st.subheader("📍 Location Map")
                        fig = create_map(location, lat, lon, result["is_fake"])
                        st.plotly_chart(fig, use_container_width=True)
                
                # Save to history
                st.session_state["analysis_history"].append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tweet": tweet[:50] + "...",
                    "result": result,
                    "location": location
                })
    else:
        st.warning("Please enter a tweet")

# History
if st.session_state["analysis_history"]:
    st.markdown("---")
    st.subheader("📚 Recent")
    for item in reversed(st.session_state["analysis_history"][-5:]):
        emoji = "❌" if item["result"]["is_fake"] else "✅"
        st.caption(f"{emoji} {item['timestamp']} - {item['tweet']}")

# Footer
st.markdown("---")
st.caption("🚀 Hybrid BERT + Mock Analysis")
