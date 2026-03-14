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
.metric-card {
    background: linear-gradient(135deg, #667eea20, #764ba220);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #667eea40;
    text-align: center;
    margin: 5px;
}
.measurement-panel {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">'
    "<h1>🚨 AI Fake Disaster Tweet Detector</h1>"
    "<p>Advanced Analytics & Measurement System</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ================================================================
# ENHANCED SESSION STATE WITH MEASUREMENTS
# ================================================================
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []
if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = ""
if "analysis_cache" not in st.session_state:
    st.session_state["analysis_cache"] = {}
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "hybrid"
if "performance_metrics" not in st.session_state:
    st.session_state["performance_metrics"] = {
        "response_times": deque(maxlen=50),
        "bert_confidence": deque(maxlen=50),
        "mock_confidence": deque(maxlen=50),
        "disaster_types": {},
        "locations_detected": {},
        "daily_analyses": {},
        "model_usage": {"bert": 0, "mock": 0, "hybrid": 0}
    }

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

# Fake news patterns
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
# BERT MODEL LOADING (with fallback)
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

# ================================================================
# ENHANCED MOCK ANALYSIS WITH MEASUREMENTS
# ================================================================

def analyze_mock(text):
    """Enhanced mock analysis with detailed measurements"""
    start_time = time.time()
    text_lower = text.lower()
    
    # Initialize detailed metrics
    fake_score = 0
    real_score = 0
    fake_details = {category: 0 for category in FAKE_PATTERNS}
    real_details = {category: 0 for category in REAL_PATTERNS}
    detected_fake_words = []
    detected_real_words = []
    
    # Check fake patterns
    for category, patterns in FAKE_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                fake_score += 1
                fake_details[category] += 1
                detected_fake_words.append(pattern)
    
    # Check real patterns
    for category, patterns in REAL_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                real_score += 1.5
                real_details[category] += 1
                detected_real_words.append(pattern)
    
    # Detect disaster type
    detected_disasters = []
    for disaster, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_disasters.append(disaster)
    
    # Word analysis
    words = text.split()
    word_count = len(words)
    unique_words = len(set(words))
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    
    # Sentence analysis
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Punctuation analysis
    exclamation_count = text.count('!')
    question_count = text.count('?')
    period_count = text.count('.')
    comma_count = text.count(',')
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
    caps_ratio = caps_words / word_count if word_count > 0 else 0
    
    # URL and number detection
    has_url = bool(re.search(r'http[s]?://', text_lower))
    numbers = re.findall(r'\d+', text)
    number_count = len(numbers)
    
    # Calculate probabilities
    total = fake_score + real_score
    if total > 0:
        fake_prob = fake_score / total
    else:
        fake_prob = 0.5
    
    real_prob = 1 - fake_prob
    
    # Calculate confidence metrics
    fake_confidence = fake_score / (fake_score + 1) if fake_score > 0 else 0
    real_confidence = real_score / (real_score + 1) if real_score > 0 else 0
    overall_confidence = abs(fake_prob - 0.5) * 2
    
    # Calculate readability score (simple Flesch-like metric)
    readability = max(0, min(100, 100 - (avg_word_length * 10) + (sentence_count * 2)))
    
    # Calculate sensationalism score
    sensationalism = min(1.0, (fake_details.get("sensational", 0) * 0.3 + 
                               exclamation_count * 0.1 + 
                               caps_ratio * 0.5))
    
    # Calculate credibility score
    credibility = min(1.0, (real_details.get("sources", 0) * 0.3 + 
                            real_details.get("measured", 0) * 0.3 + 
                            (1 if has_url else 0) * 0.2))
    
    response_time = time.time() - start_time
    
    return {
        "is_fake": fake_prob > 0.5,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "overall_confidence": overall_confidence,
        "fake_confidence": fake_confidence,
        "real_confidence": real_confidence,
        "fake_score": fake_score,
        "real_score": real_score,
        "fake_details": fake_details,
        "real_details": real_details,
        "detected_fake_words": list(set(detected_fake_words))[:10],
        "detected_real_words": list(set(detected_real_words))[:10],
        "detected_disasters": detected_disasters,
        "word_count": word_count,
        "unique_words": unique_words,
        "avg_word_length": avg_word_length,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "period_count": period_count,
        "comma_count": comma_count,
        "caps_words": caps_words,
        "caps_ratio": caps_ratio,
        "has_url": has_url,
        "number_count": number_count,
        "readability": readability,
        "sensationalism": sensationalism,
        "credibility": credibility,
        "response_time": response_time,
        "model": "Mock"
    }

# ================================================================
# BERT ANALYSIS (if available)
# ================================================================

def analyze_bert(text):
    """BERT analysis with enhanced measurements"""
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
        
        # Normalize
        total = fake_prob + real_prob
        fake_prob = fake_prob / total
        real_prob = real_prob / total
        
        # Calculate entropy (uncertainty)
        entropy = - (fake_prob * np.log2(fake_prob + 1e-10) + 
                    real_prob * np.log2(real_prob + 1e-10))
        
        response_time = time.time() - start_time
        
        return {
            "is_fake": fake_prob > 0.5,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": abs(fake_prob - 0.5) * 2,
            "entropy": entropy,
            "raw_scores": probs.tolist(),
            "response_time": response_time,
            "model": "BERT"
        }
    except Exception as e:
        return None

# ================================================================
# HYBRID ANALYSIS WITH ENHANCED MEASUREMENTS
# ================================================================

def analyze_hybrid(text):
    """Enhanced hybrid analysis with comprehensive measurements"""
    mock_result = analyze_mock(text)
    bert_result = analyze_bert(text) if bert_loaded else None
    
    start_time = time.time()
    
    if bert_result:
        # Weighted average
        fake_prob = (bert_result["fake_probability"] * 0.6 + 
                    mock_result["fake_probability"] * 0.4)
        
        # Calculate agreement between models
        agreement = 1 - abs(bert_result["fake_probability"] - mock_result["fake_probability"])
        
        # Combined confidence
        combined_confidence = (bert_result.get("confidence", 0) * 0.6 + 
                              mock_result["overall_confidence"] * 0.4)
        
        result = {
            "is_fake": fake_prob > 0.5,
            "fake_probability": fake_prob,
            "real_probability": 1 - fake_prob,
            "combined_confidence": combined_confidence,
            "model_agreement": agreement,
            "bert_contribution": bert_result["fake_probability"],
            "mock_contribution": mock_result["fake_probability"],
            "bert_confidence": bert_result.get("confidence", 0),
            "mock_confidence": mock_result["overall_confidence"],
            "entropy": bert_result.get("entropy", 0),
            **{k: v for k, v in mock_result.items() 
               if k not in ["is_fake", "fake_probability", "real_probability", "model"]},
            "model": "Hybrid (BERT+Mock)" if bert_loaded else "Mock Only",
            "bert_available": bert_loaded,
            "response_time": (bert_result["response_time"] + mock_result["response_time"]) / 2
        }
    else:
        result = mock_result
        result["model"] = "Mock Only (BERT unavailable)"
        result["response_time"] = mock_result["response_time"]
    
    result["total_processing_time"] = time.time() - start_time
    return result

# ================================================================
# MEASUREMENT VISUALIZATION FUNCTIONS
# ================================================================

def display_comprehensive_metrics(analysis):
    """Display all analysis metrics in organized panels"""
    
    st.markdown("### 📊 Comprehensive Analysis Metrics")
    
    # Probability Overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Fake Probability</h4>
            <h2 style="color: {'#ff4444' if analysis['fake_probability'] > 0.5 else '#888'}">
                {analysis['fake_probability']*100:.1f}%
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Real Probability</h4>
            <h2 style="color: {'#00cc66' if analysis['real_probability'] > 0.5 else '#888'}">
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
    
    # Confidence Metrics
    st.markdown("### 🎯 Confidence Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        conf = analysis.get('overall_confidence', analysis.get('combined_confidence', 0.5))
        st.metric("Overall Confidence", f"{conf*100:.1f}%", 
                 help="How certain the model is")
    
    with col2:
        if 'model_agreement' in analysis:
            st.metric("Model Agreement", f"{analysis['model_agreement']*100:.1f}%",
                     help="How much BERT and Mock agree")
    
    with col3:
        if 'entropy' in analysis:
            st.metric("Uncertainty", f"{analysis['entropy']:.3f}",
                     help="Lower = more certain")
    
    with col4:
        if 'response_time' in analysis:
            st.metric("Response Time", f"{analysis['response_time']*1000:.0f}ms")
    
    # Text Analysis Metrics
    st.markdown("### 📝 Text Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Words", analysis.get('word_count', 0))
        st.metric("Unique Words", analysis.get('unique_words', 0))
    
    with col2:
        st.metric("Sentences", analysis.get('sentence_count', 0))
        st.metric("Avg Word Length", f"{analysis.get('avg_word_length', 0):.1f}")
    
    with col3:
        st.metric("Readability", f"{analysis.get('readability', 0):.0f}",
                 help="Higher = easier to read")
        st.metric("Caps Ratio", f"{analysis.get('caps_ratio', 0)*100:.0f}%")
    
    with col4:
        st.metric("Exclamation", analysis.get('exclamation_count', 0))
        st.metric("Questions", analysis.get('question_count', 0))
    
    # Indicator Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚨 Fake News Indicators")
        if analysis.get('fake_details'):
            for category, count in analysis['fake_details'].items():
                if count > 0:
                    st.warning(f"**{category.title()}**: {count} indicators")
        
        if analysis.get('detected_fake_words'):
            st.caption(f"Words: {', '.join(analysis['detected_fake_words'][:5])}")
    
    with col2:
        st.markdown("#### ✅ Real News Indicators")
        if analysis.get('real_details'):
            for category, count in analysis['real_details'].items():
                if count > 0:
                    st.success(f"**{category.title()}**: {count} indicators")
        
        if analysis.get('detected_real_words'):
            st.caption(f"Words: {', '.join(analysis['detected_real_words'][:5])}")
    
    # Disaster Detection
    if analysis.get('detected_disasters'):
        st.markdown("#### 🌪️ Detected Disaster Types")
        cols = st.columns(len(analysis['detected_disasters']))
        for i, disaster in enumerate(analysis['detected_disasters']):
            with cols[i]:
                st.info(disaster.upper())
    
    # URL and Numbers
    if analysis.get('has_url'):
        st.info("🔗 Contains reference link")
    
    if analysis.get('number_count', 0) > 0:
        st.info(f"🔢 Contains {analysis['number_count']} numbers")

def display_performance_dashboard():
    """Display performance metrics dashboard"""
    
    st.markdown("### 📈 Performance Dashboard")
    
    metrics = st.session_state["performance_metrics"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_response = mean(metrics["response_times"]) if metrics["response_times"] else 0
        st.metric("Avg Response", f"{avg_response*1000:.0f}ms")
    
    with col2:
        total = len(st.session_state["analysis_history"])
        st.metric("Total Analyses", total)
    
    with col3:
        if metrics["bert_confidence"]:
            avg_bert_conf = mean(metrics["bert_confidence"]) * 100
            st.metric("Avg BERT Conf", f"{avg_bert_conf:.1f}%")
    
    with col4:
        if metrics["mock_confidence"]:
            avg_mock_conf = mean(metrics["mock_confidence"]) * 100
            st.metric("Avg Mock Conf", f"{avg_mock_conf:.1f}%")
    
    # Model Usage Pie Chart
    if sum(metrics["model_usage"].values()) > 0:
        fig = px.pie(
            values=list(metrics["model_usage"].values()),
            names=list(metrics["model_usage"].keys()),
            title="Model Usage Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Response Time Trend
    if len(metrics["response_times"]) > 1:
        fig = px.line(
            y=list(metrics["response_times"]),
            title="Response Time Trend",
            labels={"index": "Analysis #", "value": "Seconds"}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_location_stats(location, analysis):
    """Display location-based statistics"""
    
    st.markdown("### 📍 Location Analysis")
    
    # Update location stats
    perf_metrics = st.session_state["performance_metrics"]
    perf_metrics["locations_detected"][location] = perf_metrics["locations_detected"].get(location, 0) + 1
    
    # Get coordinates
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location)}&format=json&limit=1"
        headers = {'User-Agent': 'Disaster-Detector/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        
        if data:
            lat, lon = float(data[0]['lat']), float(data[0]['lon'])
            
            # Create enhanced map
            fig = go.Figure()
            
            # Add marker
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='red' if analysis['is_fake'] else 'green',
                    symbol='marker'
                ),
                text=[location],
                textposition="top center"
            ))
            
            # Add 5km radius circle
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
                line=dict(width=2, color='red' if analysis['is_fake'] else 'green'),
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
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Location stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latitude", f"{lat:.4f}°")
            with col2:
                st.metric("Longitude", f"{lon:.4f}°")
            with col3:
                st.metric("Times Detected", 
                         perf_metrics["locations_detected"][location])
            
    except Exception as e:
        st.warning(f"Could not load map for {location}")

# ================================================================
# SIDEBAR WITH ENHANCED STATISTICS
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
    
    # Real-time Statistics
    st.subheader("📈 Live Statistics")
    
    metrics = st.session_state["performance_metrics"]
    history = st.session_state["analysis_history"]
    
    if history:
        fake_count = sum(1 for h in history if h["result"]["is_fake"])
        real_count = len(history) - fake_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fake Detected", fake_count)
        with col2:
            st.metric("Real Detected", real_count)
        
        if metrics["response_times"]:
            avg_time = mean(metrics["response_times"]) * 1000
            st.metric("Avg Response", f"{avg_time:.0f}ms")
    
    st.metric("Total Analyses", len(history))
    
    # Performance dashboard expander
    with st.expander("📊 Performance Dashboard"):
        display_performance_dashboard()
    
    if st.button("🗑️ Clear All Data"):
        st.session_state["analysis_history"] = []
        st.session_state["analysis_cache"] = {}
        st.session_state["performance_metrics"] = {
            "response_times": deque(maxlen=50),
            "bert_confidence": deque(maxlen=50),
            "mock_confidence": deque(maxlen=50),
            "disaster_types": {},
            "locations_detected": {},
            "daily_analyses": {},
            "model_usage": {"bert": 0, "mock": 0, "hybrid": 0}
        }
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
if st.button("🔍 Analyze with Advanced Metrics", type="primary", use_container_width=True):
    if tweet:
        with st.spinner("Analyzing with comprehensive metrics..."):
            # Choose analysis method
            if st.session_state["model_choice"] == "bert" and bert_loaded:
                result = analyze_bert(tweet)
                st.session_state["performance_metrics"]["model_usage"]["bert"] += 1
            elif st.session_state["model_choice"] == "mock":
                result = analyze_mock(tweet)
                st.session_state["performance_metrics"]["model_usage"]["mock"] += 1
            else:  # hybrid
                result = analyze_hybrid(tweet)
                st.session_state["performance_metrics"]["model_usage"]["hybrid"] += 1
            
            if result:
                # Update performance metrics
                st.session_state["performance_metrics"]["response_times"].append(
                    result.get("response_time", 0)
                )
                
                if "overall_confidence" in result:
                    st.session_state["performance_metrics"]["mock_confidence"].append(
                        result["overall_confidence"]
                    )
                
                if "bert_confidence" in result:
                    st.session_state["performance_metrics"]["bert_confidence"].append(
                        result["bert_confidence"]
                    )
                
                # Display results
                st.markdown("---")
                
                # Main alert
                if result["is_fake"]:
                    st.markdown(
                        f'<div class="fake-alert">❌ FAKE NEWS DETECTED<br>'
                        f'Confidence: {result.get("overall_confidence", result.get("combined_confidence", 0.5))*100:.1f}%</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="real-alert">✅ REAL NEWS<br>'
                        f'Confidence: {result.get("overall_confidence", result.get("combined_confidence", 0.5))*100:.1f}%</div>',
                        unsafe_allow_html=True
                    )
                
                # Display comprehensive metrics
                display_comprehensive_metrics(result)
                
                # Location detection
                location = None
                for loc in MALAYSIA_LOCATIONS:
                    if loc.lower() in tweet.lower():
                        location = loc
                        break
                
                if location:
                    display_location_stats(location, result)
                
                # Model info
                st.info(f"🤖 Model: {result.get('model', 'Unknown')}")
                
                # Save to history
                st.session_state["analysis_history"].append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tweet": tweet[:50] + "..." if len(tweet) > 50 else tweet,
                    "result": result,
                    "location": location
                })
                
                # Update disaster types
                for disaster in result.get("detected_disasters", []):
                    st.session_state["performance_metrics"]["disaster_types"][disaster] = \
                        st.session_state["performance_metrics"]["disaster_types"].get(disaster, 0) + 1
    else:
        st.warning("Please enter a tweet")

# Enhanced History Section
if st.session_state["analysis_history"]:
    st.markdown("---")
    st.subheader("📚 Analysis History")
    
    # Create DataFrame for history
    history_data = []
    for item in st.session_state["analysis_history"]:
        history_data.append({
            "Time": item["timestamp"],
            "Tweet": item["tweet"],
            "Fake %": f"{item['result']['fake_probability']*100:.1f}%",
            "Real %": f"{item['result']['real_probability']*100:.1f}%",
            "Decision": "❌ FAKE" if item['result']['is_fake'] else "✅ REAL",
            "Location": item.get("location", "Unknown"),
            "Model": item['result'].get('model', 'Unknown')
        })
    
    df = pd.DataFrame(history_data[::-1])  # Reverse for latest first
    st.dataframe(df, use_container_width=True)
    
    # Disaster type distribution
    if st.session_state["performance_metrics"]["disaster_types"]:
        st.subheader("🌪️ Disaster Type Distribution")
        fig = px.bar(
            x=list(st.session_state["performance_metrics"]["disaster_types"].keys()),
            y=list(st.session_state["performance_metrics"]["disaster_types"].values()),
            labels={"x": "Disaster Type", "y": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer with summary stats
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Analyses", len(st.session_state["analysis_history"]))
with col2:
    fake_total = sum(1 for h in st.session_state["analysis_history"] if h["result"]["is_fake"])
    st.metric("Total Fake", fake_total)
with col3:
    real_total = len(st.session_state["analysis_history"]) - fake_total
    st.metric("Total Real", real_total)
with col4:
    if st.session_state["performance_metrics"]["response_times"]:
        avg_resp = mean(st.session_state["performance_metrics"]["response_times"]) * 1000
        st.metric("Avg Response", f"{avg_resp:.0f}ms")

st.caption("🚀 Advanced Analytics - Every tweet measured comprehensively")
