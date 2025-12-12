"""
JanSpandana.AI - Complete Enhanced Voice Feedback Collection App
FIXES ALL ISSUES:
1. ✅ Proper AI voice response playback  
2. ✅ Simplified flow - no redundant forms
3. ✅ AI insights display (confidence, sentiment, processing time)
4. ✅ Enhanced error handling with fallback notifications
5. ✅ Admin dashboard for AI monitoring
6. ✅ Voice-first natural conversation flow
7. ✅ Mobile-optimized interface

CLEAN USER EXPERIENCE:
- Single "Start Conversation" button
- AI handles name/village collection naturally  
- True voice conversation (user speaks, AI responds with voice)
- Real-time AI insights for transparency
"""

import base64
import io
import json
import os
import queue
import threading
import time
from datetime import datetime
import requests
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# Page configuration
st.set_page_config(
    page_title="JanSpandana.AI - Voice Feedback",
    page_icon=":speaking_head_in_silhouette:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Enhanced CSS with AI insights styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@300;400;600;700&display=swap');

.main-title {
    font-family: 'Noto Sans Telugu', sans-serif;
    font-size: 3rem;
    color: #1e40af;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 700;
}

.subtitle {
    font-family: 'Noto Sans Telugu', sans-serif;
    font-size: 1.3rem;
    color: #374151;
    text-align: center;
    margin-bottom: 2rem;
    line-height: 1.6;
}

.instruction {
    font-family: 'Noto Sans Telugu', sans-serif;
    font-size: 1.1rem;
    background: #f0f9ff;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
}

.conversation-box {
    background: #f8fafc;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #e2e8f0;
}

.user-message {
    background: #dbeafe;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #2563eb;
}

.ai-message {
    background: #dcfce7;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #16a34a;
}

.telugu-text {
    font-family: 'Noto Sans Telugu', sans-serif;
    font-size: 1.1rem;
    line-height: 1.8;
}

.ai-insights {
    background: #fef7cd;
    padding: 0.8rem;
    border-radius: 6px;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    border-left: 3px solid #f59e0b;
}

.confidence-high { color: #059669; font-weight: bold; }
.confidence-medium { color: #d97706; font-weight: bold; }
.confidence-low { color: #dc2626; font-weight: bold; }

.processing-time {
    color: #6b7280;
    font-size: 0.8rem;
    font-style: italic;
}

.fallback-notice {
    background: #fef2f2;
    color: #991b1b;
    padding: 0.8rem;
    border-radius: 6px;
    font-size: 0.9rem;
    border-left: 3px solid #f87171;
}

.admin-dashboard {
    background: #f3f4f6;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid #d1d5db;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.village-info {
    background: #f0f9ff;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.audio-player {
    margin: 1rem 0;
    width: 100%;
}
</style>
""",
    unsafe_allow_html=True,
)

class EnhancedJanSpandanaApp:
    """Enhanced feedback collection app with AI insights and voice fixes"""

    def __init__(self):
        # Allow overriding API base via environment variable if backend is on a different host/port
        self.api_base = os.getenv("API_BASE_URL", "http://localhost:8000")
        # Allow configurable request timeout (defaults to 60s to avoid long AI processing timeouts)
        self.request_timeout = float(os.getenv("API_TIMEOUT_SECONDS", "60"))
        self.health_timeout = float(os.getenv("API_HEALTH_TIMEOUT_SECONDS", "15"))

        # Initialize session state
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = None
        if "conversation_started" not in st.session_state:
            st.session_state.conversation_started = False
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "user_name" not in st.session_state:
            st.session_state.user_name = ""
        if "village_name" not in st.session_state:
            st.session_state.village_name = ""
        if "current_stage" not in st.session_state:
            st.session_state.current_stage = "welcome"
        if "ai_insights_history" not in st.session_state:
            st.session_state.ai_insights_history = []
        if "show_admin" not in st.session_state:
            st.session_state.show_admin = False

    def check_backend_health(self) -> dict:
        """Check backend and AI services health"""
        try:
            health_response = requests.get(f"{self.api_base}/health", timeout=self.health_timeout)
            if health_response.status_code < 500:
                try:
                    data = health_response.json()
                except Exception:
                    # Response came back but wasn't JSON
                    return {"status": "error", "services": {}, "reachable": True}
                data["reachable"] = True
                return data
            return {"status": "error", "services": {}, "reachable": False}
        except Exception:
            return {"status": "offline", "services": {}, "reachable": False}

    def start_conversation(self) -> dict:
        """Start new conversation with simplified approach"""
        try:
            payload = {
                "metadata": {
                    "source": "streamlit_enhanced_app",
                    "timestamp": datetime.now().isoformat(),
                    "ui_version": "2.0_enhanced"
                }
            }

            response = requests.post(
                f"{self.api_base}/api/v1/conversation/start", 
                json=payload, 
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def send_text_message(self, conversation_id: str, message: str) -> dict:
        """Send text message with enhanced response handling"""
        try:
            payload = {"conversation_id": conversation_id, "user_input": message}

            response = requests.post(
                f"{self.api_base}/api/v1/conversation/text", 
                json=payload, 
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def send_voice_message(self, conversation_id: str, audio_data: bytes) -> dict:
        """Send voice message with enhanced response handling"""
        try:
            files = {"audio_file": ("recording.webm", io.BytesIO(audio_data), "audio/webm")}
            data = {
                "conversation_id": conversation_id, 
                "audio_format": "webm", 
                "region_hint": "coastal"
            }

            response = requests.post(
                f"{self.api_base}/api/v1/conversation/voice", 
                files=files, 
                data=data, 
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def end_conversation(self, conversation_id: str) -> dict:
        """End conversation"""
        try:
            payload = {
                "conversation_id": conversation_id, 
                "completion_reason": "user_completed"
            }

            response = requests.post(
                f"{self.api_base}/api/v1/conversation/end", 
                json=payload, 
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def get_admin_stats(self) -> dict:
        """Get administrative statistics"""
        try:
            response = requests.get(f"{self.api_base}/api/v1/admin/stats", timeout=self.request_timeout)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}

    def get_ai_insights(self) -> dict:
        """Get AI performance insights"""
        try:
            response = requests.get(f"{self.api_base}/api/v1/admin/ai-insights", timeout=self.request_timeout)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}

    def display_ai_insights(self, ai_insights: dict, processing_stats: dict = None):
        """Display AI insights in user-friendly format"""
        if not ai_insights:
            return

        # Create insights container
        with st.expander("🤖 AI Insights", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                confidence = ai_insights.get("confidence_score", 0)
                if confidence >= 0.8:
                    st.markdown(f'<span class="confidence-high">Confidence: {confidence:.0%}</span>', unsafe_allow_html=True)
                elif confidence >= 0.6:
                    st.markdown(f'<span class="confidence-medium">Confidence: {confidence:.0%}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="confidence-low">Confidence: {confidence:.0%}</span>', unsafe_allow_html=True)
            
            with col2:
                processing_time = ai_insights.get("processing_time_ms", 0)
                st.markdown(f'<span class="processing-time">Processing: {processing_time:.0f}ms</span>', unsafe_allow_html=True)
            
            with col3:
                fallback_used = ai_insights.get("fallback_used", False)
                if fallback_used:
                    st.markdown('<span style="color: #dc2626;">⚠️ Fallback Mode</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color: #059669;">✅ AI Powered</span>', unsafe_allow_html=True)
            
            # Additional insights
            if ai_insights.get("reasoning"):
                st.markdown("**AI Understanding:**")
                st.markdown(f'<div class="ai-insights">{ai_insights["reasoning"]}</div>', unsafe_allow_html=True)
            
            if ai_insights.get("sentiment_detected"):
                sentiment = ai_insights["sentiment_detected"]
                sentiment_emoji = {"concerned": "😟", "frustrated": "😤", "hopeful": "😊", "satisfied": "😌"}.get(sentiment, "😐")
                st.markdown(f"**Sentiment Detected:** {sentiment_emoji} {sentiment.title()}")
            
            if ai_insights.get("urgency_level"):
                urgency = ai_insights["urgency_level"]
                urgency_color = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}
                st.markdown(f"**Urgency:** {urgency_color.get(urgency, '⚪')} {urgency.title()}")

    def play_ai_audio_response(self, audio_url: str):
        """Play AI audio response with proper URL construction"""
        if audio_url:
            # Construct full URL if relative path
            if audio_url.startswith("/static/"):
                full_audio_url = f"{self.api_base}{audio_url}"
            else:
                full_audio_url = audio_url
            
            try:
                # Display audio player
                st.audio(full_audio_url, format="audio/mp3")
                return True
            except Exception as e:
                st.warning(f"Could not play audio: {str(e)}")
                return False
        return False

    def show_fallback_notice(self, error_message: str):
        """Show fallback notice when AI fails"""
        st.markdown(
            f"""
            <div class="fallback-notice">
                <strong>⚠️ AI Fallback Mode:</strong> {error_message}<br>
                The system is using simplified processing. Voice responses may be limited.
            </div>
            """,
            unsafe_allow_html=True
        )

    def show_admin_dashboard(self):
        """Display admin dashboard with AI monitoring"""
        st.markdown("## 🔧 Admin Dashboard")
        
        # Get admin stats
        admin_stats = self.get_admin_stats()
        ai_insights = self.get_ai_insights()
        
        if admin_stats:
            st.markdown("### System Status")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Active Conversations", admin_stats.get("active_conversations", 0))
            with col2:
                ai_conversations = admin_stats.get("ai_powered_conversations", 0)
                st.metric("AI Powered", ai_conversations)
            with col3:
                ai_metrics = admin_stats.get("ai_metrics", {})
                success_rate = ai_metrics.get("ai_success_rate", 0)
                st.metric("AI Success Rate", f"{success_rate:.1%}")
            with col4:
                avg_time = ai_metrics.get("avg_ai_processing_time_ms", 0)
                st.metric("Avg AI Time", f"{avg_time:.0f}ms")
        
        # AI Service Health
        if ai_insights:
            st.markdown("### AI Service Health")
            ai_status = ai_insights.get("ai_service_status", {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Core Services:**")
                for service, status in ai_status.items():
                    if isinstance(status, bool):
                        icon = "✅" if status else "❌"
                        st.markdown(f"{icon} {service.replace('_', ' ').title()}")
            
            with col2:
                if "performance_metrics" in ai_insights:
                    metrics = ai_insights["performance_metrics"]
                    st.markdown("**Performance Metrics:**")
                    st.json(metrics)
        
        # Recent Performance
        if ai_insights and "recent_performance" in ai_insights:
            st.markdown("### Recent AI Performance")
            recent_perf = ai_insights["recent_performance"]
            if recent_perf:
                # Simple performance visualization
                success_count = sum(1 for p in recent_perf if p.get("success", False))
                total_count = len(recent_perf)
                if total_count > 0:
                    st.progress(success_count / total_count)
                    st.caption(f"Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")

# Initialize app
app = EnhancedJanSpandanaApp()

# Header
st.markdown(
    '<h1 class="main-title">JanSpandana.AI Voice Assistant</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Share your grievance naturally using voice or text. '
    "AI will understand and help forward your concerns to the right authorities.</p>",
    unsafe_allow_html=True,
)

# Admin toggle (hidden feature)
if st.sidebar.button("🔧 Admin Dashboard"):
    st.session_state.show_admin = not st.session_state.show_admin

# Check backend health
health_status = app.check_backend_health()
backend_online = health_status.get("reachable", False) or health_status.get("status") not in ["offline", None]

if not backend_online:
    st.warning("⚠️ Backend is not reachable right now. UI will stay available; retry after backend responds.")
    st.code("python main.py")
    if st.button("Retry backend health"):
        st.experimental_rerun()

# Show AI service status
if health_status.get("status") == "ai_degraded":
    st.warning("⚠️ AI services are partially functional. Some features may use simplified processing.")
elif health_status.get("status") == "ai_fallback":
    st.warning("⚠️ Running in fallback mode. AI features limited.")
elif health_status.get("status") == "error":
    st.warning("⚠️ Backend reachable but reported service errors (database/speech/AI).")

# Admin Dashboard
if st.session_state.show_admin:
    app.show_admin_dashboard()
    st.markdown("---")

# Main Interface
if st.session_state.current_stage == "welcome":
    st.markdown(
        """
        <div class="instruction">
            <h3>🎤 Natural Voice Conversation</h3>
            <p>Click "Start Conversation" below and talk naturally with our AI assistant.</p>
            <p><strong>What to expect:</strong></p>
            <ul>
                <li>AI will greet you and ask your name</li>
                <li>Tell us your village and the type of issue you're facing</li>
                <li>Describe your problem in detail - AI will ask follow-up questions</li>
                <li>Your feedback will be forwarded to relevant authorities</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Single start button - no forms!
    if st.button("🚀 Start Conversation", key="start_conversation", use_container_width=True):
        with st.spinner("Starting conversation with AI..."):
            result = app.start_conversation()

            if result.get("success"):
                st.session_state.conversation_id = result["conversation_id"]
                st.session_state.conversation_started = True
                st.session_state.current_stage = "conversation"

                # Get initial AI response
                initial_text = result.get("initial_text", "Welcome! How can I help you?")
                ai_insights = result.get("ai_insights", {})

                st.session_state.messages.append(
                    {
                        "type": "ai",
                        "content": initial_text,
                        "timestamp": datetime.now(),
                        "ai_insights": ai_insights
                    }
                )

                # Play initial audio if available
                initial_audio_url = result.get("initial_audio_url")
                if initial_audio_url:
                    app.play_ai_audio_response(initial_audio_url)

                st.success("✅ Conversation started! AI is ready to listen.")
                st.rerun()
            else:
                error_msg = result.get('error', 'Unknown error')
                st.error(f"❌ Could not start conversation: {error_msg}")
                app.show_fallback_notice(error_msg)

# Conversation Interface
elif st.session_state.current_stage == "conversation":
    st.markdown(
        f"""
        <div class="village-info">
            <strong>🗣️ Conversation ID:</strong> {st.session_state.conversation_id}<br>
            <strong>💬 Messages:</strong> {len(st.session_state.messages)}<br>
            <strong>🤖 AI Status:</strong> {health_status.get("status", "unknown").title()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display conversation history
    if st.session_state.messages:
        st.markdown("### 💬 Conversation")

        for message in st.session_state.messages:
            if message["type"] == "user":
                st.markdown(
                    f"""
                    <div class="user-message">
                        <strong>👤 You:</strong><br>
                        <span class="telugu-text">{message["content"]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="ai-message">
                        <strong>🤖 JanSpandana AI:</strong><br>
                        <span class="telugu-text">{message["content"]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Display AI insights for this message
                if message.get("ai_insights"):
                    app.display_ai_insights(
                        message["ai_insights"], 
                        message.get("processing_stats")
                    )

    st.markdown("### 💬 Send a Message")

    # Text input
    col_input1, col_input2 = st.columns([3, 1])

    with col_input1:
        user_message = st.text_area(
            "Type your message (Telugu or English):",
            placeholder="Example: నా గ్రామంలో వైద్య సేవలు బాగా లేవు (Medical services are poor in my village)",
            height=100,
        )

    with col_input2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📤 Send Text", key="send_text", use_container_width=True):
            if user_message.strip():
                with st.spinner("🤖 AI is thinking..."):
                    # Add user message
                    st.session_state.messages.append(
                        {
                            "type": "user",
                            "content": user_message,
                            "timestamp": datetime.now(),
                        }
                    )

                    # Send to AI
                    result = app.send_text_message(st.session_state.conversation_id, user_message)

                    if result.get("success"):
                        ai_response_text = result.get("ai_response_text", "No response received.")
                        ai_insights = result.get("ai_insights", {})
                        processing_stats = result.get("processing_stats", {})

                        # Add AI response
                        st.session_state.messages.append(
                            {
                                "type": "ai",
                                "content": ai_response_text,
                                "timestamp": datetime.now(),
                                "ai_insights": ai_insights,
                                "processing_stats": processing_stats
                            }
                        )

                        # Play AI audio response
                        ai_audio_url = result.get("ai_response_audio_url")
                        if ai_audio_url:
                            st.audio(f"{app.api_base}{ai_audio_url}", format="audio/mp3")

                        st.rerun()
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        st.error(f"❌ Unable to send message: {error_msg}")
                        app.show_fallback_notice(error_msg)
            else:
                st.warning("⚠️ Message cannot be empty.")

    st.markdown("---")

    # Voice input
    st.markdown("### 🎤 Record Voice Message")

    # Use dynamic key to prevent recorder issues
    recorder_key = f"voice_recorder_{len(st.session_state.messages)}_{st.session_state.conversation_id}"
    
    audio_recording = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording", 
        key=recorder_key,
        use_container_width=True,
    )

    if audio_recording and audio_recording.get("bytes"):
        audio_bytes = audio_recording["bytes"]
        
        # Show audio player
        st.audio(audio_bytes, format="audio/wav")

        if st.button("🚀 Send Voice Message", key="send_voice", use_container_width=True):
            with st.spinner("🎧 Processing your voice..."):
                result = app.send_voice_message(st.session_state.conversation_id, audio_bytes)

                if result.get("success"):
                    # User transcript
                    user_transcript = result.get("user_transcript", "Transcript not available.")
                    
                    st.session_state.messages.append(
                        {
                            "type": "user",
                            "content": f"🎤 {user_transcript}",
                            "timestamp": datetime.now(),
                        }
                    )

                    # AI response
                    ai_response_text = result.get("ai_response_text", "No response received.")
                    ai_insights = result.get("ai_insights", {})
                    processing_stats = result.get("processing_stats", {})

                    st.session_state.messages.append(
                        {
                            "type": "ai",
                            "content": ai_response_text,
                            "timestamp": datetime.now(),
                            "ai_insights": ai_insights,
                            "processing_stats": processing_stats
                        }
                    )

                    # 🔥 FIX: Play AI voice response
                    ai_audio_url = result.get("ai_response_audio_url")
                    if ai_audio_url:
                        # Construct full URL and play audio
                        full_audio_url = f"{app.api_base}{ai_audio_url}"
                        st.audio(full_audio_url, format="audio/mp3")
                        st.success("🔊 AI responded with voice!")
                    else:
                        st.info("🔇 AI response (text only)")

                    st.success("✅ Voice message sent successfully!")
                    st.rerun()
                else:
                    error_msg = result.get('error', 'Unknown error')
                    st.error(f"❌ Unable to process voice: {error_msg}")
                    app.show_fallback_notice(error_msg)

    st.markdown("---")
    
    # Control buttons
    col_control1, col_control2 = st.columns(2)

    with col_control1:
        if st.button("🔄 Start Over", key="new_conversation", use_container_width=True):
            if st.session_state.conversation_id:
                app.end_conversation(st.session_state.conversation_id)
            
            # Reset session state
            for key in ["conversation_id", "conversation_started", "messages", "user_name", "village_name"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_stage = "welcome"
            st.rerun()

    with col_control2:
        if st.button("✅ Finish Conversation", key="end_conversation", use_container_width=True):
            with st.spinner("Finishing conversation..."):
                result = app.end_conversation(st.session_state.conversation_id)

                if result.get("success"):
                    st.session_state.current_stage = "completed"
                    
                    # Show final AI insights if available
                    if "ai_insights" in result:
                        st.success("✅ Conversation completed successfully!")
                        app.display_ai_insights(result["ai_insights"])
                    
                    st.rerun()
                else:
                    st.error("❌ Could not end conversation properly.")

# Completion stage
elif st.session_state.current_stage == "completed":
    st.success("🎉 Thank you for sharing your feedback!")

    st.markdown(
        """
        <div class="instruction">
            <h3>✅ Your Voice Has Been Heard</h3>
            <p>Your conversation has been successfully processed and will be reviewed by relevant authorities.</p>
            <p><strong>What happens next:</strong></p>
            <ul>
                <li>🔍 AI has analyzed your feedback for priority and department routing</li>
                <li>📋 Your concerns have been categorized and logged in the system</li>
                <li>👥 Relevant government departments will receive your feedback</li>
                <li>📞 Follow-up contact may be made if you provided contact information</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Show conversation summary if available
    if st.session_state.ai_insights_history:
        st.markdown("### 📊 Conversation Summary")
        latest_insights = st.session_state.ai_insights_history[-1]
        app.display_ai_insights(latest_insights)

    if st.button("🆕 Start New Feedback Session", use_container_width=True):
        # Complete reset
        for key in ["conversation_id", "conversation_started", "messages", "user_name", "village_name", "current_stage", "ai_insights_history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p><strong>JanSpandana.AI v2.0 Enhanced</strong> | AI-Powered Voice Feedback System</p>
        <p>Backend Status: {health_status.get("status", "unknown").title()} | 
        AI Services: {"🤖 Active" if health_status.get("ai_services", {}).get("conversation_engine") else "⚡ Fallback"}</p>
        <p>Support: support@janspandana.gov.in | Made with ❤️ for rural Andhra Pradesh</p>
    </div>
    """,
    unsafe_allow_html=True,
)
