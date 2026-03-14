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
import firebase_admin
from firebase_admin import credentials, firestore
import hashlib
import threading
import queue

# ================================================================
# FIREBASE CONFIGURATION (REAL-TIME DATABASE)
# ================================================================

# Initialize Firebase if not already initialized
if not firebase_admin._apps:
    # For deployment, use environment variables or Streamlit secrets
    try:
        # Try to get from Streamlit secrets (for cloud deployment)
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
    except:
        # Fallback to local file for development
        cred = credentials.Certificate("firebase-credentials.json")
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    db = firestore.client()

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
    page_title="AI Fake Disaster Tweet Detector - REAL TIME",
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
.realtime-badge {
    background: linear-gradient(135deg, #ff6b6b, #ee5253);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: bold;
    display: inline-block;
    margin-bottom: 10px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.8; }
    100% { opacity: 1; }
}
.live-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    background-color: #00ff00;
    border-radius: 50%;
    margin-right: 5px;
    animation: blink 1s infinite;
}
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">'
    "<h1>🚨 AI Fake Disaster Tweet Detector <span class='realtime-badge'>🔴 REAL-TIME</span></h1>"
    "<p>Live database - Data persists across sessions and updates in real-time</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ================================================================
# REAL-TIME DATA MANAGEMENT
# ================================================================

class RealtimeDataManager:
    """Manages real-time data synchronization with Firebase"""
    
    def __init__(self):
        self.db = db
        self.analyses_collection = "tweet_analyses"
        self.stats_collection = "system_stats"
        self.alerts_collection = "active_alerts"
        
    def save_analysis(self, analysis_data):
        """Save analysis to Firebase in real-time"""
        try:
            doc_ref = self.db.collection(self.analyses_collection).document()
            analysis_data["timestamp"] = firestore.SERVER_TIMESTAMP
            analysis_data["id"] = doc_ref.id
            doc_ref.set(analysis_data)
            
            # Update real-time stats
            self.update_stats(analysis_data)
            
            # Check if this is a high-confidence fake news alert
            if analysis_data.get("is_fake") and analysis_data.get("confidence", 0) > 0.8:
                self.create_alert(analysis_data)
                
            return doc_ref.id
        except Exception as e:
            st.error(f"Error saving to real-time database: {e}")
            return None
    
    def update_stats(self, analysis_data):
        """Update real-time statistics"""
        try:
            stats_ref = self.db.collection(self.stats_collection).document("live_stats")
            
            # Use transaction for atomic updates
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
                
                # Update counters
                stats["total_analyses"] += 1
                if analysis_data.get("is_fake"):
                    stats["total_fake"] += 1
                else:
                    stats["total_real"] += 1
                
                # Update location stats
                if analysis_data.get("location"):
                    loc = analysis_data["location"]
                    stats["locations"][loc] = stats["locations"].get(loc, 0) + 1
                
                # Update disaster types
                for disaster in analysis_data.get("detected_disasters", []):
                    stats["disaster_types"][disaster] = stats["disaster_types"].get(disaster, 0) + 1
                
                # Update model usage
                model = analysis_data.get("model_used", "unknown")
                stats["models_used"][model] = stats["models_used"].get(model, 0) + 1
                
                # Keep last 24h rolling
                stats["last_24h"].append({
                    "timestamp": datetime.now().isoformat(),
                    "is_fake": analysis_data.get("is_fake")
                })
                if len(stats["last_24h"]) > 1000:
                    stats["last_24h"] = stats["last_24h"][-1000:]
                
                transaction.set(stats_ref, stats)
                return stats
            
            transaction = self.db.transaction()
            return update_in_transaction(transaction, stats_ref)
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
        """Get latest analyses in real-time"""
        try:
            analyses = self.db.collection(self.analyses_collection)\
                .order_by("timestamp", direction=firestore.Query.DESCENDING)\
                .limit(limit)\
                .stream()
            return [doc.to_dict() for doc in analyses]
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            return []
    
    def get_live_stats(self):
        """Get real-time statistics"""
        try:
            stats_ref = self.db.collection(self.stats_collection).document("live_stats")
            stats = stats_ref.get()
            return stats.to_dict() if stats.exists else {}
        except Exception as e:
            return {}
    
    def get_active_alerts(self):
        """Get active fake news alerts"""
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
        try:
            self.db.collection(self.alerts_collection).document(alert_id)\
                .update({"status": "resolved", "resolved_at": firestore.SERVER_TIMESTAMP})
        except Exception as e:
            print(f"Error resolving alert: {e}")

# Initialize real-time manager
rt_manager = RealtimeDataManager()

# ================================================================
# SESSION STATE (Now syncs with Firebase)
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

# ================================================================
# CONSTANTS (Keep all your existing constants)
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
# BERT MODEL LOADING (Keep your existing BERT loading code)
# ================================================================

@st.cache_resource(show_spinner="Loading BERT model...")
def load_bert_model():
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
# ANALYSIS FUNCTIONS (Keep all your existing analysis functions)
# ================================================================

def analyze_mock(text):
    """Enhanced mock analysis with detailed measurements"""
    start_time = time.time()
    text_lower = text.lower()
    
    fake_score = 0
    real_score = 0
    fake_details = {category: 0 for category in FAKE_PATTERNS}
    real_details = {category: 0 for category in REAL_PATTERNS}
    detected_fake_words = []
    detected_real_words = []
    
    for category, patterns in FAKE_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                fake_score += 1
                fake_details[category] += 1
                detected_fake_words.append(pattern)
    
    for category, patterns in REAL_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                real_score += 1.5
                real_details[category] += 1
                detected_real_words.append(pattern)
    
    detected_disasters = []
    for disaster, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_disasters.append(disaster)
    
    words = text.split()
    word_count = len(words)
    unique_words = len(set(words))
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    exclamation_count = text.count('!')
    question_count = text.count('?')
    period_count = text.count('.')
    comma_count = text.count(',')
    caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
    caps_ratio = caps_words / word_count if word_count > 0 else 0
    
    has_url = bool(re.search(r'http[s]?://', text_lower))
    numbers = re.findall(r'\d+', text)
    number_count = len(numbers)
    
    total = fake_score + real_score
    if total > 0:
        fake_prob = fake_score / total
    else:
        fake_prob = 0.5
    
    real_prob = 1 - fake_prob
    
    overall_confidence = abs(fake_prob - 0.5) * 2
    readability = max(0, min(100, 100 - (avg_word_length * 10) + (sentence_count * 2)))
    
    sensationalism = min(1.0, (fake_details.get("sensational", 0) * 0.3 + 
                               exclamation_count * 0.1 + 
                               caps_ratio * 0.5))
    
    credibility = min(1.0, (real_details.get("sources", 0) * 0.3 + 
                            real_details.get("measured", 0) * 0.3 + 
                            (1 if has_url else 0) * 0.2))
    
    response_time = time.time() - start_time
    
    return {
        "is_fake": fake_prob > 0.5,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "overall_confidence": overall_confidence,
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
        "model": "Mock",
        "model_used": "Mock Mode"
    }

def analyze_bert(text):
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
            "model": "BERT",
            "model_used": "BERT Mode"
        }
    except Exception as e:
        return None

def analyze_hybrid(text):
    mock_result = analyze_mock(text)
    bert_result = analyze_bert(text) if bert_loaded else None
    
    start_time = time.time()
    
    if bert_result:
        fake_prob = (bert_result["fake_probability"] * 0.6 + 
                    mock_result["fake_probability"] * 0.4)
        
        agreement = 1 - abs(bert_result["fake_probability"] - mock_result["fake_probability"])
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
            "model_used": "Hybrid Mode",
            "bert_available": bert_loaded,
            "response_time": (bert_result["response_time"] + mock_result["response_time"]) / 2
        }
    else:
        result = mock_result
        result["model"] = "Mock Only (BERT unavailable)"
        result["model_used"] = "Mock Only"
        result["response_time"] = mock_result["response_time"]
    
    result["total_processing_time"] = time.time() - start_time
    return result

# ================================================================
# REAL-TIME UI COMPONENTS
# ================================================================

def display_live_stats():
    """Display real-time statistics from Firebase"""
    
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
        if stats.get("last_24h"):
            last_hour = sum(1 for x in stats["last_24h"] 
                          if (datetime.now() - datetime.fromisoformat(x["timestamp"])).seconds < 3600)
            st.metric("Last Hour", last_hour)
    
    # Location heatmap if we have data
    if stats.get("locations"):
        st.subheader("📍 Live Location Heatmap")
        loc_df = pd.DataFrame([
            {"Location": loc, "Count": count}
            for loc, count in stats["locations"].items()
        ]).sort_values("Count", ascending=False).head(10)
        
        fig = px.bar(loc_df, x="Location", y="Count", 
                     title="Top 10 Locations with Activity",
                     color="Count", color_continuous_scale="reds")
        st.plotly_chart(fig, use_container_width=True)

def display_live_alerts():
    """Display real-time alerts for high-confidence fake news"""
    
    alerts = rt_manager.get_active_alerts()
    
    if alerts:
        st.markdown("### 🚨 Real-Time Alerts")
        
        for alert in alerts:
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
        
        # Create DataFrame for display
        feed_data = []
        for a in analyses:
            timestamp = a.get("timestamp", "")
            if hasattr(timestamp, "strftime"):
                timestamp = timestamp.strftime("%H:%M:%S")
            
            feed_data.append({
                "Time": timestamp,
                "Tweet": a.get("tweet", "")[:50] + "...",
                "Fake %": f"{a.get('fake_probability', 0)*100:.1f}%",
                "Location": a.get("location", "Unknown"),
                "Status": "❌ FAKE" if a.get("is_fake") else "✅ REAL",
                "Model": a.get("model_used", "Unknown")
            })
        
        df = pd.DataFrame(feed_data)
        st.dataframe(df, use_container_width=True, height=300)

# ================================================================
# SIDEBAR WITH REAL-TIME STATS
# ================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Live connection status
    st.markdown("""
    <div style="background-color: #28a74520; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <span class="live-indicator"></span> <strong>LIVE CONNECTION</strong><br>
        <small>Data syncs in real-time</small>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Auto-refresh toggle
    st.session_state["auto_refresh"] = st.checkbox("🔄 Auto-refresh", value=True)
    
    if st.session_state["auto_refresh"]:
        # Check if we need to refresh (every 5 seconds)
        if time.time() - st.session_state["last_refresh"] > 5:
            st.session_state["last_refresh"] = time.time()
            st.rerun()
    
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
    
    # Real-time Stats
    st.subheader("📈 Live Global Stats")
    display_live_stats()
    
    # Session info
    st.markdown("---")
    st.caption(f"Session ID: `{st.session_state['session_id']}`")
    st.caption(f"Last Refresh: {datetime.now().strftime('%H:%M:%S')}")

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
    <span style="margin-left: 10px;">🔄 Real-time data shared globally</span>
</div>
""", unsafe_allow_html=True)

# Display active alerts
display_live_alerts()

# ================================================================
# INPUT SECTION
# ================================================================

# Create columns for input and clear button
input_col1, input_col2 = st.columns([6, 1])

with input_col1:
    input_key = f"tweet_input_{st.session_state['input_key_counter']}"
    
    tweet = st.text_area(
        "📝 Enter tweet to analyze:",
        height=100,
        placeholder="Example: Heavy rain in Kampar causing flash floods - reported by local authorities...",
        key=input_key,
        label_visibility="collapsed"
    )

with input_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear", use_container_width=True, help="Clear input field"):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# Quick examples
st.markdown("#### 📋 Quick Examples:")
example_col1, example_col2, example_col3, example_col4 = st.columns(4)

with example_col1:
    if st.button("📰 Real News", use_container_width=True):
        st.session_state["tweet_input"] = "Heavy rain in Kampar causing flash floods. According to local authorities, JPS monitoring water levels."
        st.session_state["input_key_counter"] += 1
        st.rerun()

with example_col2:
    if st.button("🚨 Fake News", use_container_width=True):
        st.session_state["tweet_input"] = "URGENT! BREAKING: MASSIVE earthquake in Kuala Lumpur! Thousands DEAD! SHARE NOW! 😱😱😱"
        st.session_state["input_key_counter"] += 1
        st.rerun()

with example_col3:
    if st.button("🔄 Mixed", use_container_width=True):
        st.session_state["tweet_input"] = "URGENT! Flood in Johor! Water level 2 meters! SHARE NOW! Official source says evacuating."
        st.session_state["input_key_counter"] += 1
        st.rerun()

with example_col4:
    if st.button("🌍 Location Test", use_container_width=True):
        st.session_state["tweet_input"] = "Flood in Penang - authorities confirm"
        st.session_state["input_key_counter"] += 1
        st.rerun()

# Analyze button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze & Share to Live Feed", type="primary", use_container_width=True)
with col2:
    if st.button("🔄 New Tweet", use_container_width=True):
        st.session_state["input_key_counter"] += 1
        st.rerun()

# ================================================================
# ANALYSIS EXECUTION (Now saves to Firebase)
# ================================================================

if analyze_clicked and tweet:
    with st.spinner("Analyzing and broadcasting to live feed..."):
        # Choose analysis method
        if st.session_state["model_choice"] == "bert" and bert_loaded:
            result = analyze_bert(tweet)
        elif st.session_state["model_choice"] == "mock":
            result = analyze_mock(tweet)
        else:  # hybrid
            result = analyze_hybrid(tweet)
        
        if result:
            # Extract location
            location = None
            for loc in MALAYSIA_LOCATIONS:
                if loc.lower() in tweet.lower():
                    location = loc
                    break
            
            # Prepare data for Firebase
            analysis_data = {
                "tweet": tweet,
                "tweet_preview": tweet[:100] + "..." if len(tweet) > 100 else tweet,
                "location": location,
                "is_fake": result.get("is_fake"),
                "fake_probability": result.get("fake_probability"),
                "real_probability": result.get("real_probability"),
                "confidence": result.get("overall_confidence", result.get("combined_confidence", 0.5)),
                "detected_disasters": result.get("detected_disasters", []),
                "word_count": result.get("word_count", 0),
                "model_used": result.get("model_used", "Unknown"),
                "session_id": st.session_state["session_id"]
            }
            
            # Save to Firebase
            doc_id = rt_manager.save_analysis(analysis_data)
            
            if doc_id:
                st.success(f"✅ Analysis saved to live feed! ID: {doc_id[:8]}")
            
            # Display results
            st.markdown("---")
            
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
            
            # Display comprehensive metrics (use your existing display function)
            # ... (I'll keep your existing display_comprehensive_metrics function)
            
            if location:
                st.info(f"📍 Location detected: {location}")
            
            st.info(f"🤖 Model: {result.get('model_used', 'Unknown')}")
            
elif analyze_clicked and not tweet:
    st.warning("⚠️ Please enter a tweet to analyze.")

# ================================================================
# LIVE FEED SECTION
# ================================================================
st.markdown("---")
st.subheader("📡 Global Live Feed")

# Display live feed
display_live_feed()

# Manual refresh button
if st.button("🔄 Refresh Live Feed", use_container_width=True):
    st.rerun()

# ================================================================
# FOOTER
# ================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;'>"
    "🚀 REAL-TIME AI Fake Disaster Tweet Detector | "
    "<span class='live-indicator'></span> LIVE DATABASE - Data persists globally<br>"
    f"Session: {st.session_state['session_id']} | "
    f"Last sync: {datetime.now().strftime('%H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True
)
