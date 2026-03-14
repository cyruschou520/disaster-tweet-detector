import streamlit as st
import pandas as pd
import torch
import requests
import urllib.parse
import time
import json
import re
import math
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from functools import wraps
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
    animation: pulse 2s infinite;
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
@keyframes pulse {
    0%  { opacity: 1.0; }
    50% { opacity: 0.75; }
    100%{ opacity: 1.0; }
}
.tweet-guidelines {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin-bottom: 20px;
}
.tweet-example {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 5px;
    border: 1px solid #ddd;
    margin: 5px 0;
    font-family: monospace;
}
.metric-card {
    background: linear-gradient(135deg, #667eea20, #764ba220);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #667eea40;
    text-align: center;
    margin: 5px;
}
.free-badge {
    background: linear-gradient(135deg, #28a745, #20c997);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7em;
    margin-left: 5px;
    display: inline-block;
}
.bert-badge {
    background: linear-gradient(135deg, #667eea, #5a67d8);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7em;
    margin-left: 5px;
    display: inline-block;
}
.mock-badge {
    background: linear-gradient(135deg, #48bb78, #38a169);
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.7em;
    margin-left: 5px;
    display: inline-block;
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
    font-size: 14px;
}
.real-bar {
    height: 100%;
    background: linear-gradient(90deg, #00cc66, #008844);
    color: white;
    text-align: center;
    line-height: 30px;
    font-weight: bold;
    font-size: 14px;
}
.model-indicator {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: bold;
    margin-right: 10px;
}
.bert-indicator {
    background: linear-gradient(135deg, #667eea20, #5a67d820);
    border: 2px solid #667eea;
    color: #667eea;
}
.mock-indicator {
    background: linear-gradient(135deg, #48bb7820, #38a16920);
    border: 2px solid #48bb78;
    color: #48bb78;
}
.hybrid-indicator {
    background: linear-gradient(135deg, #ffaa0020, #ff880020);
    border: 2px solid #ffaa00;
    color: #ffaa00;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">'
    "<h1>🚨 AI Fake Disaster Tweet Detector <span class='bert-badge'>BERT + MOCK</span></h1>"
    "<p>Hybrid detection system combining BERT model with keyword analysis</p>"
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
if "map_style" not in st.session_state:
    st.session_state["map_style"] = "Street Map"
if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 10
if "analysis_count" not in st.session_state:
    st.session_state["analysis_count"] = 0
if "widget_key_counter" not in st.session_state:
    st.session_state["widget_key_counter"] = 0
if "analysis_cache" not in st.session_state:
    st.session_state["analysis_cache"] = {}
if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "hybrid"  # hybrid, bert, mock

# ================================================================
# CONSTANTS
# ================================================================
MALAYSIA_LOCATIONS = [
    'Kampar', 'Ipoh', 'Kuala Lumpur', 'KL', 'Penang', 'Pulau Pinang',
    'Johor', 'Johor Bahru', 'Shah Alam', 'Selangor', 'Perak', 'Kedah',
    'Kelantan', 'Terengganu', 'Pahang', 'Negeri Sembilan', 'Melaka',
    'Malacca', 'Perlis', 'Sabah', 'Sarawak', 'Langkawi', 'Kuantan',
    'Kota Bharu', 'Alor Setar', 'George Town', 'Butterworth', 'Taiping',
    'Teluk Intan', 'Petaling Jaya', 'Subang Jaya', 'Klang', 'Putrajaya',
    'Cameron Highlands', 'Genting Highlands', 'Port Dickson', 'Miri',
    'Kota Kinabalu', 'Sandakan', 'Tawau', 'Sibu', 'Kuching'
]

# Fake news keywords and patterns (for mock mode)
FAKE_NEWS_PATTERNS = {
    "urgency_markers": {
        "keywords": [
            'urgent', 'breaking', 'just in', 'alert', 'warning', 'immediate',
            'segera', 'penting', 'perhatian', 'awas', 'bahaya', '!!!', '???',
            'now', 'asap', 'right now', 'immediately', 'tonight', 'today'
        ],
        "weight": 1.0
    },
    "sensational_words": {
        "keywords": [
            'unbelievable', 'shocking', 'horrific', 'terrible', 'devastating',
            'tragic', 'teruk', 'ngeri', 'dahsyat', 'mengerikan', 'massive',
            'giant', 'huge', 'enormous', 'catastrophic', 'apocalyptic',
            'worst ever', 'never seen', 'unprecedented', 'historical'
        ],
        "weight": 1.2
    },
    "vague_language": {
        "keywords": [
            'they say', 'people are saying', 'rumors', 'heard that', 'allegedly',
            'kata orang', 'dengar cerita', 'khabar angin', 'kononnya', 'supposedly',
            'apparently', 'might have', 'could be', 'perhaps', 'maybe'
        ],
        "weight": 1.1
    },
    "sharing_requests": {
        "keywords": [
            'share', 'spread', 'forward', 'viral', 'retweet', 'repost',
            'kongsi', 'sebarkan', 'viral kan', 'share now', 'spread the word',
            'tell everyone', 'pass it on', 'go viral', 'like and share'
        ],
        "weight": 1.3
    },
    "conspiracy_indicators": {
        "keywords": [
            'they don\'t want you to know', 'government hiding', 'cover up',
            'secret', 'hidden truth', 'censored', 'banned', 'suppressed',
            'they are lying', 'mainstream media won\'t tell you'
        ],
        "weight": 1.4
    },
    "emotional_language": {
        "keywords": [
            'pray', 'please help', 'god', 'omg', 'oh my god', 'pity',
            '可怜', 'help them', 'save them', 'tragedy', 'heartbreaking',
            '哭', '伤心', '难过', 'sad', 'crying', 'terrified', 'scared'
        ],
        "weight": 0.8
    }
}

REAL_NEWS_PATTERNS = {
    "sources_mentioned": {
        "keywords": [
            'according to', 'reported by', 'official', 'authorities',
            'police', 'jabatan bomba', 'bomba', 'JPS', 'MET Malaysia',
            'met malaysia', 'jabatan meteorologi', 'kkmm', 'nadma',
            'local authorities', 'government', 'minister', 'department',
            'statement from', 'press release', 'confirmed by', 'verified'
        ],
        "weight": 1.5
    },
    "specific_details": {
        "keywords": [
            'at', 'on', 'date', 'time', 'location', 'coordinates',
            'reported at', 'occurred at', 'happened at', 'measured',
            'recorded', 'observed', 'detected', 'sensor', 'station'
        ],
        "weight": 1.2
    },
    "official_terms": {
        "keywords": [
            'investigation', 'monitoring', 'assessment', 'response',
            'evacuation', 'relief', 'aid', 'assistance', 'coordination',
            'update', 'advisory', 'warning', 'alert', 'notice'
        ],
        "weight": 1.1
    },
    "measured_data": {
        "keywords": [
            'mm', 'cm', 'meter', 'km', 'celsius', 'degrees', 'magnitude',
            'speed', 'velocity', 'level', 'height', 'depth', 'volume',
            'number of', 'total', 'count', 'estimated'
        ],
        "weight": 1.3
    }
}

# ================================================================
# LOAD BERT MODEL
# ================================================================

@st.cache_resource
def load_bert_model():
    """Load BERT model for fake news detection"""
    try:
        # Try to load fine-tuned model if exists
        model = AutoModelForSequenceClassification.from_pretrained("bert_model")
        tokenizer = AutoTokenizer.from_pretrained("bert_model")
        st.sidebar.success("✅ BERT model loaded successfully")
        return model, tokenizer, True
    except:
        try:
            # Fallback to base model for zero-shot
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            st.sidebar.warning("⚠️ Using base BERT model (not fine-tuned)")
            return model, tokenizer, True
        except:
            st.sidebar.error("❌ BERT model not available - using Mock Mode only")
            return None, None, False

bert_model, bert_tokenizer, bert_available = load_bert_model()

# ================================================================
# CALLBACK FUNCTIONS
# ================================================================

def clear_input_callback():
    """Callback function to clear the input"""
    st.session_state["tweet_input"] = ""
    st.session_state["widget_key_counter"] += 1

# ================================================================
# BERT ANALYSIS FUNCTION
# ================================================================

def analyze_with_bert(text):
    """Analyze tweet using BERT model"""
    if not bert_available or bert_model is None or bert_tokenizer is None:
        return None
    
    try:
        inputs = bert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
        
        # Get probabilities
        probs = torch.softmax(outputs.logits, dim=1)[0]
        fake_prob = probs[1].item()  # Assuming label 1 is fake
        real_prob = probs[0].item()   # Assuming label 0 is real
        
        # Normalize to sum to 1.0
        total = fake_prob + real_prob
        fake_prob = fake_prob / total
        real_prob = real_prob / total
        
        # Determine if fake
        is_fake = fake_prob > 0.5
        
        # Calculate confidence
        confidence = abs(fake_prob - 0.5) * 2
        
        # Generate reasons based on BERT's confidence
        reasons = []
        if is_fake:
            reasons.append(f"🤖 BERT model detected fake news patterns")
            if confidence > 0.8:
                reasons.append("🔴 High confidence fake news detection")
            elif confidence > 0.6:
                reasons.append("🟡 Moderate confidence fake news detection")
        else:
            reasons.append(f"🤖 BERT model classified as credible")
            if confidence > 0.8:
                reasons.append("🟢 High confidence real news detection")
        
        return {
            "is_fake": is_fake,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": confidence,
            "reasons": reasons,
            "model_used": "BERT",
            "response_time": 0.5  # Approximate
        }
    except Exception as e:
        st.error(f"BERT analysis error: {e}")
        return None

# ================================================================
# MOCK ANALYSIS FUNCTION (DETERMINISTIC)
# ================================================================

def analyze_with_mock(text):
    """
    Deterministic mock analysis using keyword detection
    """
    text_lower = text.lower()
    
    # Initialize scoring
    fake_score = 0
    real_score = 0
    total_weight = 0
    
    # Track found indicators
    found_fake_indicators = {
        "urgency_markers": [],
        "sensational_words": [],
        "vague_language": [],
        "sharing_requests": [],
        "conspiracy_indicators": [],
        "emotional_language": []
    }
    
    found_real_indicators = {
        "sources_mentioned": [],
        "specific_details": [],
        "official_terms": [],
        "measured_data": []
    }
    
    # Check fake news patterns (weighted)
    for category, data in FAKE_NEWS_PATTERNS.items():
        for pattern in data["keywords"]:
            if pattern in text_lower:
                if category in found_fake_indicators:
                    found_fake_indicators[category].append(pattern)
                fake_score += data["weight"]
                total_weight += data["weight"]
    
    # Check real news patterns (weighted)
    for category, data in REAL_NEWS_PATTERNS.items():
        for pattern in data["keywords"]:
            if pattern in text_lower:
                if category in found_real_indicators:
                    found_real_indicators[category].append(pattern)
                real_score += data["weight"]
                total_weight += data["weight"]
    
    # Length and structure analysis
    words = text.split()
    word_count = len(words)
    
    if word_count > 50:
        real_score += 2.0
        total_weight += 2.0
    elif word_count < 10:
        fake_score += 1.0
        total_weight += 1.0
    
    # ALL CAPS detection
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
    if caps_words > 2:
        fake_score += caps_words * 0.3
        total_weight += caps_words * 0.3
    
    # Exclamation marks
    exclamation_count = text.count('!')
    if exclamation_count > 2:
        fake_score += min(exclamation_count * 0.2, 1.0)
        total_weight += min(exclamation_count * 0.2, 1.0)
    
    # URLs
    has_url = bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', text_lower))
    if has_url:
        real_score += 1.0
        total_weight += 1.0
    
    # Numbers
    numbers = re.findall(r'\d+', text)
    if len(numbers) > 2:
        real_score += min(len(numbers) * 0.2, 1.0)
        total_weight += min(len(numbers) * 0.2, 1.0)
    
    # Calculate probabilities
    if total_weight < 0.1:
        total_weight = 0.1
        fake_score = 0.05
        real_score = 0.05
    
    raw_fake = fake_score / total_weight
    raw_real = real_score / total_weight
    
    total_raw = raw_fake + raw_real
    fake_probability = raw_fake / total_raw
    real_probability = raw_real / total_raw
    
    is_fake = fake_probability > 0.5
    confidence = abs(fake_probability - 0.5) * 2
    
    # Calculate additional scores
    sensationalism_score = len(found_fake_indicators["sensational_words"]) / 8
    sensationalism_score = min(sensationalism_score + (exclamation_count / 15), 1.0)
    
    credibility_score = len(found_real_indicators["sources_mentioned"]) / 5
    credibility_score += len(found_real_indicators["measured_data"]) / 6
    credibility_score = min(credibility_score, 1.0)
    
    # Generate reasons
    reasons = []
    if is_fake:
        if found_fake_indicators["urgency_markers"]:
            reasons.append(f"🚨 Urgency markers: {', '.join(found_fake_indicators['urgency_markers'][:3])}")
        if found_fake_indicators["sensational_words"]:
            reasons.append(f"📢 Sensational language: {', '.join(found_fake_indicators['sensational_words'][:3])}")
        if found_fake_indicators["sharing_requests"]:
            reasons.append(f"🔄 Sharing requests: {', '.join(found_fake_indicators['sharing_requests'][:3])}")
    else:
        if found_real_indicators["sources_mentioned"]:
            reasons.append(f"📰 Official sources: {', '.join(found_real_indicators['sources_mentioned'][:3])}")
        if found_real_indicators["measured_data"]:
            reasons.append(f"📊 Specific data: {', '.join(found_real_indicators['measured_data'][:3])}")
    
    if not reasons:
        reasons.append("No strong indicators found - analysis based on general patterns")
    
    # Determine urgency
    urgency_score = len(found_fake_indicators["urgency_markers"]) + exclamation_count/4
    urgency_level = "critical" if urgency_score > 4 else "high" if urgency_score > 2.5 else "medium" if urgency_score > 1 else "low"
    
    # Credibility indicators
    credibility_indicators = {
        "has_sources": len(found_real_indicators["sources_mentioned"]) > 0,
        "specific_details": len(found_real_indicators["measured_data"]) > 0 or len(numbers) > 2,
        "official_language": len(found_real_indicators["official_terms"]) > 0,
        "sensationalism": len(found_fake_indicators["sensational_words"]) > 0 or exclamation_count > 3,
        "calls_to_share": len(found_fake_indicators["sharing_requests"]) > 0,
        "emotional_language": len(found_fake_indicators["emotional_language"]) > 0 or caps_words > 2
    }
    
    return {
        "is_fake": is_fake,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "confidence": confidence,
        "reasons": reasons[:5],
        "sensationalism_score": sensationalism_score,
        "credibility_score": credibility_score,
        "word_count": word_count,
        "exclamation_count": exclamation_count,
        "caps_words": caps_words,
        "has_url": has_url,
        "numbers_found": len(numbers),
        "credibility_indicators": credibility_indicators,
        "fake_indicators": found_fake_indicators,
        "real_indicators": found_real_indicators,
        "urgency_level": urgency_level,
        "model_used": "Mock Mode",
        "response_time": 0.3
    }

# ================================================================
# HYBRID ANALYSIS FUNCTION
# ================================================================

def analyze_with_hybrid(text):
    """
    Combine BERT and Mock mode for more accurate detection
    """
    # Get results from both models
    bert_result = analyze_with_bert(text) if bert_available else None
    mock_result = analyze_with_mock(text)
    
    if bert_result:
        # Weighted average (BERT 60%, Mock 40%)
        fake_prob = (bert_result["fake_probability"] * 0.6 + 
                    mock_result["fake_probability"] * 0.4)
        real_prob = 1 - fake_prob
        
        # Combine reasons
        reasons = []
        reasons.append("🤖 BERT Analysis:")
        reasons.extend(bert_result["reasons"])
        reasons.append("")
        reasons.append("🔍 Keyword Analysis:")
        reasons.extend(mock_result["reasons"])
        
        # Determine if fake
        is_fake = fake_prob > 0.5
        confidence = abs(fake_prob - 0.5) * 2
        
        # Combine other metrics
        result = {
            "is_fake": is_fake,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": confidence,
            "reasons": reasons[:7],
            "sensationalism_score": mock_result["sensationalism_score"],
            "credibility_score": mock_result["credibility_score"],
            "word_count": mock_result["word_count"],
            "exclamation_count": mock_result["exclamation_count"],
            "caps_words": mock_result["caps_words"],
            "has_url": mock_result["has_url"],
            "numbers_found": mock_result["numbers_found"],
            "credibility_indicators": mock_result["credibility_indicators"],
            "fake_indicators": mock_result["fake_indicators"],
            "real_indicators": mock_result["real_indicators"],
            "urgency_level": mock_result["urgency_level"],
            "model_used": "Hybrid (BERT + Mock)",
            "bert_contribution": bert_result["fake_probability"],
            "mock_contribution": mock_result["fake_probability"],
            "response_time": (bert_result.get("response_time", 0.5) + 
                            mock_result["response_time"]) / 2
        }
    else:
        # Fallback to mock only
        result = mock_result
        result["model_used"] = "Mock Only (BERT unavailable)"
    
    return result

# ================================================================
# MAIN ANALYSIS FUNCTION
# ================================================================

def analyze_tweet(text, model_choice):
    """Main analysis function that routes to selected model"""
    
    # Check cache first
    cache_key = f"{model_choice}_{text}"
    if cache_key in st.session_state["analysis_cache"]:
        return st.session_state["analysis_cache"][cache_key]
    
    # Route to appropriate model
    if model_choice == "bert" and bert_available:
        result = analyze_with_bert(text)
    elif model_choice == "mock":
        result = analyze_with_mock(text)
    else:  # hybrid or fallback
        result = analyze_with_hybrid(text)
    
    # Cache result if valid
    if result:
        st.session_state["analysis_cache"][cache_key] = result
    
    return result

# ================================================================
# PROBABILITY VISUALIZATION
# ================================================================

def display_probability_bar(fake_prob, real_prob):
    """Display a visual probability bar"""
    fake_percent = fake_prob * 100
    real_percent = real_prob * 100
    
    st.markdown(f"""
    <div style="margin: 20px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="color: #ff4444; font-weight: bold;">FAKE: {fake_percent:.1f}%</span>
            <span style="color: #00cc66; font-weight: bold;">REAL: {real_percent:.1f}%</span>
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
# LOCATION FUNCTIONS
# ================================================================

def extract_location(text):
    """Extract Malaysia location from tweet text"""
    text_lower = text.lower()
    for loc in MALAYSIA_LOCATIONS:
        if loc.lower() in text_lower:
            return loc
    return None

def get_location_details(location: str):
    """Get comprehensive location details from OpenStreetMap"""
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location)}&format=json&limit=1&addressdetails=1"
        headers = {'User-Agent': 'AI-Disaster-Monitoring-System/1.0'}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        if data and len(data) > 0:
            result = data[0]
            return {
                'lat': float(result.get('lat', 0)),
                'lon': float(result.get('lon', 0)),
                'display_name': result.get('display_name', location),
                'address': result.get('address', {}),
                'type': result.get('type', 'unknown')
            }
    except Exception as e:
        st.warning(f"Could not fetch location details: {e}")
    return None

def create_location_map(location_data, is_fake=False):
    """Create a simple location map"""
    if not location_data:
        return None
    
    fig = go.Figure()
    
    marker_color = 'red' if is_fake else 'green'
    
    fig.add_trace(go.Scattermapbox(
        lat=[location_data['lat']],
        lon=[location_data['lon']],
        mode='markers+text',
        marker=dict(size=20, color=marker_color, symbol='marker'),
        text=[location_data.get('display_name', 'Location').split(',')[0]],
        textposition="top center",
        hoverinfo='text'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=location_data['lat'], lon=location_data['lon']),
            zoom=10
        ),
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=400
    )
    
    return fig

# ================================================================
# METRICS DISPLAY FUNCTIONS
# ================================================================

def display_analysis_metrics(analysis):
    """Display detailed metrics from the analysis"""
    
    st.markdown("### 📊 Analysis Metrics")
    display_probability_bar(
        analysis.get('fake_probability', 0.5),
        analysis.get('real_probability', 0.5)
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fake_prob = analysis.get('fake_probability', 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>Fake Probability</h4>
            <h2 style="color: {'#ff4444' if fake_prob > 50 else '#888'}">{fake_prob:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        real_prob = analysis.get('real_probability', 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>Real Probability</h4>
            <h2 style="color: {'#00cc66' if real_prob > 50 else '#888'}">{real_prob:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    if 'sensationalism_score' in analysis:
        with col3:
            sens_score = analysis.get('sensationalism_score', 0) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h4>Sensationalism</h4>
                <h2 style="color: {'#ff4444' if sens_score > 50 else '#888'}">{sens_score:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
    
    if 'credibility_score' in analysis:
        with col4:
            cred_score = analysis.get('credibility_score', 0) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h4>Credibility</h4>
                <h2 style="color: {'#00cc66' if cred_score > 50 else '#888'}">{cred_score:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Technical metrics
    cols = st.columns(4)
    metrics = [
        ("Response Time", f"{analysis.get('response_time', 0):.2f}s"),
        ("Word Count", analysis.get('word_count', 0)),
        ("Exclamation Marks", analysis.get('exclamation_count', 0)),
        ("ALL CAPS Words", analysis.get('caps_words', 0))
    ]
    
    for i, (label, value) in enumerate(metrics):
        with cols[i]:
            st.metric(label, value)

def display_credibility_indicators(indicators):
    """Display credibility indicators in a grid"""
    if not indicators:
        return
    
    st.markdown("### 📋 Credibility Indicators")
    cols = st.columns(3)
    
    indicator_configs = [
        ("Has Sources", indicators.get("has_sources", False), "📰", "❌"),
        ("Specific Details", indicators.get("specific_details", False), "📊", "❌"),
        ("Official Language", indicators.get("official_language", False), "🏛️", "❌"),
        ("Sensationalism", indicators.get("sensationalism", False), "⚠️", "✅"),
        ("Calls to Share", indicators.get("calls_to_share", False), "🔄", "✅"),
        ("Emotional Language", indicators.get("emotional_language", False), "😢", "✅")
    ]
    
    for i, (label, value, true_icon, false_icon) in enumerate(indicator_configs):
        with cols[i % 3]:
            icon = true_icon if value else false_icon
            bg_color = "#d4edda" if value else "#f8d7da"
            text_color = "#155724" if value else "#721c24"
            st.markdown(f"""
            <div style="background-color: {bg_color}; color: {text_color}; padding: 10px; 
                        border-radius: 5px; margin: 5px; text-align: center;">
                <strong>{icon} {label}</strong>
            </div>
            """, unsafe_allow_html=True)

# ================================================================
# MAIN UI
# ================================================================

# Sidebar
with st.sidebar:
    st.header("🎛️ Model Selection")
    
    # Model selection
    model_choice = st.radio(
        "Choose Detection Model",
        options=["hybrid", "bert", "mock"],
        format_func=lambda x: {
            "hybrid": "🤖 Hybrid (BERT + Mock) - Recommended",
            "bert": "🧠 BERT Only - Deep Learning",
            "mock": "🔍 Mock Mode - Keyword Analysis"
        }.get(x, x),
        index=0,
        key="model_choice"
    )
    
    # Model status indicators
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if bert_available:
            st.markdown("🧠 **BERT:** ✅ Ready")
        else:
            st.markdown("🧠 **BERT:** ❌ Unavailable")
    with col2:
        st.markdown("🔍 **Mock:** ✅ Ready")
    
    if model_choice == "bert" and not bert_available:
        st.warning("⚠️ BERT not available - falling back to Hybrid mode")
        st.session_state["model_choice"] = "hybrid"
    
    st.markdown("---")
    
    # Statistics
    st.header("📊 Session Statistics")
    total = len(st.session_state["analysis_history"])
    fake_count = sum(1 for h in st.session_state["analysis_history"] if h.get("is_fake", False))
    real_count = total - fake_count
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("Fake", fake_count, delta_color="inverse")
    with col3:
        st.metric("Real", real_count)
    
    if total > 0:
        avg_conf = sum(h.get("confidence", 0) for h in st.session_state["analysis_history"]) / total * 100
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
    st.markdown("---")
    st.caption(f"📦 Cached analyses: {len(st.session_state['analysis_cache'])}")
    
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state["analysis_cache"] = {}
        st.rerun()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["analysis_history"] = []
        st.rerun()

# Main content
model_name = {
    "hybrid": "Hybrid (BERT + Mock)",
    "bert": "BERT Deep Learning",
    "mock": "Mock Keyword Analysis"
}.get(st.session_state["model_choice"], "Hybrid")

model_badge = {
    "hybrid": "hybrid-indicator",
    "bert": "bert-indicator",
    "mock": "mock-indicator"
}.get(st.session_state["model_choice"], "hybrid-indicator")

st.markdown(f"""
<div style="margin-bottom: 20px;">
    <span class="model-indicator {model_badge}">🤖 Active Model: {model_name}</span>
    <span class="free-badge">FREE - NO API KEY</span>
</div>
""", unsafe_allow_html=True)

# Tweet input guidelines
st.markdown("""
<div class="tweet-guidelines">
    <h4>📝 Enter a tweet to analyze for fake news:</h4>
    <p><strong>Try these examples:</strong></p>
    <table style="width:100%">
        <tr>
            <td style="width:50%">❌ <em>"URGENT! BREAKING: MASSIVE earthquake in Kuala Lumpur! Thousands DEAD! SHARE NOW! 😱😱😱"</em></td>
            <td>✅ <em>"Heavy rain in Kampar causing flash floods - reported by local authorities. JPS monitoring water levels."</em></td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# Input area
text_area_key = f"tweet_input_{st.session_state['widget_key_counter']}"
tweet = st.text_area(
    "Tweet text:",
    height=120,
    placeholder="Paste or type a tweet here...",
    key=text_area_key
)

# Buttons
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    analyze_clicked = st.button("🔍 Analyze Tweet", type="primary", use_container_width=True)
with col2:
    st.button("🔄 Clear Input", on_click=clear_input_callback, use_container_width=True)

# Store current tweet
if tweet:
    st.session_state["tweet_input"] = tweet

# Analysis execution
if analyze_clicked and st.session_state["tweet_input"]:
    with st.spinner(f"🔍 Analyzing with {model_name}..."):
        analysis = analyze_tweet(
            st.session_state["tweet_input"],
            st.session_state["model_choice"]
        )
        
        if analysis:
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Classification banner
            is_fake = analysis.get("is_fake", False)
            fake_prob = analysis.get("fake_probability", 0.5) * 100
            real_prob = analysis.get("real_probability", 0.5) * 100
            
            alert_class = "fake-alert" if is_fake else "real-alert"
            if analysis.get("model_used") == "Hybrid (BERT + Mock)":
                alert_class = "hybrid-alert"
            
            st.markdown(
                f'<div class="{alert_class}">'
                f'{"❌ FAKE" if is_fake else "✅ REAL"} NEWS DETECTED<br>'
                f'Fake: {fake_prob:.1f}% / Real: {real_prob:.1f}%<br>'
                f'<small>Model: {analysis.get("model_used", "Unknown")}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Display reasons
            if analysis.get("reasons"):
                st.markdown("### 🔍 Detection Reasons")
                for reason in analysis["reasons"]:
                    st.warning(f"• {reason}")
            
            # Display metrics
            display_analysis_metrics(analysis)
            
            # Display credibility indicators
            if analysis.get("credibility_indicators"):
                display_credibility_indicators(analysis["credibility_indicators"])
            
            # Extract and display location
            location = extract_location(st.session_state["tweet_input"])
            if location:
                with st.spinner("🗺️ Fetching location data..."):
                    location_data = get_location_details(location)
                    if location_data:
                        st.markdown("### 🗺️ Location Map")
                        fig = create_location_map(location_data, is_fake)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            # Save to history
            st.session_state["analysis_history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tweet": st.session_state["tweet_input"][:100] + "...",
                "analysis": analysis,
                "is_fake": is_fake,
                "fake_probability": analysis.get("fake_probability", 0.5),
                "real_probability": analysis.get("real_probability", 0.5),
                "location": location if location else "Unknown",
                "model": analysis.get("model_used", "Unknown")
            })
            
            st.session_state["analysis_count"] += 1

elif analyze_clicked and not st.session_state["tweet_input"]:
    st.warning("⚠️ Please enter a tweet to analyze.")

# History section
if st.session_state["analysis_history"]:
    st.markdown("---")
    st.subheader("📚 Recent Analyses")
    
    for i, item in enumerate(reversed(st.session_state["analysis_history"][-5:])):
        emoji = "❌" if item.get("is_fake", False) else "✅"
        fake_prob = item.get("fake_probability", 0.5) * 100
        real_prob = item.get("real_probability", 0.5) * 100
        
        with st.expander(
            f"{emoji} {item['timestamp']} - {item['tweet']} - "
            f"{item.get('location', 'No location')} (Fake: {fake_prob:.1f}%)"
        ):
            st.json(item['analysis'])

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;'>"
    "🚀 AI Fake Disaster Tweet Detector | "
    "<span class='bert-badge'>BERT</span> + <span class='mock-badge'>MOCK</span><br>"
    f"Total analyses: {st.session_state['analysis_count']} | "
    f"Cached: {len(st.session_state['analysis_cache'])}"
    "</div>",
    unsafe_allow_html=True
)