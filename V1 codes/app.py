"""
JanSpandana.AI - Voice Feedback Collection App
Streamlit interface for villagers to submit voice grievances
Connects to FastAPI backend for real-time AI conversation
"""

import base64
import io
import json
import queue
import threading
import time
from datetime import datetime

import av
import numpy as np
import requests
import streamlit as st
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

# Page configuration
st.set_page_config(
    page_title="JanSpandana.AI - Voice Feedback",
    page_icon=":speaking_head_in_silhouette:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Configuration
API_BASE_URL = "http://localhost:8000"
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Custom CSS for interface
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

.record-button {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    border: none;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-size: 1.2rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
    margin: 1rem 0;
}

.status-recording {
    background: #fef2f2;
    color: #991b1b;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    border: 2px dashed #f87171;
}

.status-processing {
    background: #fefbf2;
    color: #92400e;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    font-weight: 600;
    border: 2px dashed #fbbf24;
}

.village-info {
    background: #f0f9ff;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


class AudioRecorder:
    """Handle audio recording using streamlit-webrtc"""

    def __init__(self):
        self.audio_frames = []
        self.lock = threading.Lock()

    def audio_frame_callback(self, frame):
        with self.lock:
            self.audio_frames.append(frame)

    def get_audio_data(self):
        with self.lock:
            if not self.audio_frames:
                return None

            audio_data = b""
            for frame in self.audio_frames:
                for plane in frame.planes:
                    audio_data += bytes(plane)

            self.audio_frames.clear()
            return audio_data


class JanSpandanaApp:
    """Main feedback collection app"""

    def __init__(self):
        self.api_base = API_BASE_URL

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

    def start_conversation(self, village_name: str = "") -> dict:
        """Start new conversation with backend"""
        try:
            payload = {
                "village_name": village_name,
                "metadata": {"source": "streamlit_app", "timestamp": datetime.now().isoformat()},
            }

            response = requests.post(
                f"{self.api_base}/api/v1/conversation/start", json=payload, timeout=15
            )

            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:  # pragma: no cover - network call
            return {"success": False, "error": str(exc)}

    def send_text_message(self, conversation_id: str, message: str) -> dict:
        """Send text message to backend"""
        try:
            payload = {"conversation_id": conversation_id, "user_input": message}

            response = requests.post(
                f"{self.api_base}/api/v1/conversation/text", json=payload, timeout=20
            )

            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:  # pragma: no cover - network call
            return {"success": False, "error": str(exc)}

    def send_voice_message(self, conversation_id: str, audio_data: bytes) -> dict:
        """Send voice message to backend"""
        try:
            # mic_recorder returns WebM/OGG style bytes; send correct format hint
            files = {"audio_file": ("recording.webm", io.BytesIO(audio_data), "audio/webm")}
            data = {"conversation_id": conversation_id, "audio_format": "webm", "region_hint": "coastal"}

            response = requests.post(
                f"{self.api_base}/api/v1/conversation/voice", files=files, data=data, timeout=30
            )

            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:  # pragma: no cover - network call
            return {"success": False, "error": str(exc)}

    def end_conversation(self, conversation_id: str) -> dict:
        """End conversation"""
        try:
            payload = {"conversation_id": conversation_id, "completion_reason": "user_completed"}

            response = requests.post(
                f"{self.api_base}/api/v1/conversation/end", json=payload, timeout=15
            )

            if response.status_code == 200:
                return response.json()
            return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as exc:  # pragma: no cover - network call
            return {"success": False, "error": str(exc)}


# Initialize app
app = JanSpandanaApp()

# Header
st.markdown(
    '<h1 class="main-title">JanSpandana.AI Voice Assistant</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Share your grievance in Telugu or English using voice or text. '
    "We will forward it to the support team and provide an AI reply.</p>",
    unsafe_allow_html=True,
)

# Check backend connection
try:
    health_check = requests.get(f"{API_BASE_URL}/health", timeout=5)
    backend_online = health_check.status_code == 200
except Exception:
    backend_online = False

if not backend_online:
    st.error("Backend is not reachable. Please start the FastAPI server (python main.py).")
    st.code("python main.py")
    st.stop()

# Welcome section
if st.session_state.current_stage == "welcome":
    st.markdown(
        """
    <div class="instruction">
        <h3>Start your feedback</h3>
        <p>Tell us about your issue, request, or idea. You can type or speak.</p>
        <p><strong>How it works:</strong></p>
        <ul>
            <li>Enter your name and village, then start the conversation.</li>
            <li>Send a quick text or record a short voice message.</li>
            <li>Review the AI response and continue the chat as needed.</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### Share your details")

    col1, col2 = st.columns(2)

    with col1:
        user_name = st.text_input("Your name:", placeholder="Example: Lakshmi")

    with col2:
        village_name = st.text_input("Village name:", placeholder="Example: Tirupati")

    if st.button("Start conversation", key="start_conversation"):
        if user_name and village_name:
            with st.spinner("Starting conversation..."):
                result = app.start_conversation(village_name)

                if result.get("success"):
                    st.session_state.conversation_id = result["conversation_id"]
                    st.session_state.conversation_started = True
                    st.session_state.user_name = user_name
                    st.session_state.village_name = village_name
                    st.session_state.current_stage = "conversation"

                    st.session_state.messages.append(
                        {
                            "type": "ai",
                            "content": result.get(
                                "initial_text", "Welcome! Please share your concern."
                            ),
                            "timestamp": datetime.now(),
                        }
                    )

                    st.success("Conversation started.")
                    st.rerun()
                else:
                    st.error(f"Could not start conversation: {result.get('error', 'Unknown error')}")
        else:
            st.warning("Please enter both your name and village to continue.")

# Conversation interface
elif st.session_state.current_stage == "conversation":
    st.markdown(
        f"""
    <div class="village-info">
        <strong>Citizen:</strong> {st.session_state.user_name} |
        <strong>Village:</strong> {st.session_state.village_name} |
        <strong>Conversation:</strong> {st.session_state.conversation_id}
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.session_state.messages:
        st.markdown("### Conversation")

        for message in st.session_state.messages:
            if message["type"] == "user":
                st.markdown(
                    f"""
                <div class="user-message">
                    <strong>You:</strong><br>
                    <span class="telugu-text">{message["content"]}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="ai-message">
                    <strong>JanSpandana AI:</strong><br>
                    <span class="telugu-text">{message["content"]}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    st.markdown("### Send a text message")

    col_input1, col_input2 = st.columns([3, 1])

    with col_input1:
        user_message = st.text_area(
            "Type your message (Telugu or English):",
            placeholder="Example: Street lights are not working in our area.",
            height=100,
        )

    with col_input2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Send text", key="send_text"):
            if user_message.strip():
                with st.spinner("Sending to AI..."):
                    st.session_state.messages.append(
                        {
                            "type": "user",
                            "content": user_message,
                            "timestamp": datetime.now(),
                        }
                    )

                    result = app.send_text_message(st.session_state.conversation_id, user_message)

                    if result.get("success"):
                        ai_response = result.get(
                            "ai_response_text",
                            "No response received. Please try sending again.",
                        )
                        st.session_state.messages.append(
                            {
                                "type": "ai",
                                "content": ai_response,
                                "timestamp": datetime.now(),
                            }
                        )

                        st.rerun()
                    else:
                        st.error(f"Unable to send message: {result.get('error', 'Unknown error')}")
            else:
                st.warning("Message cannot be empty.")

    st.markdown("---")

    st.markdown("### Record a voice note")

    audio_recording = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        key=f"voice_recorder_{len(st.session_state.messages)}_{st.session_state.conversation_id or 'new'}",
        use_container_width=True,
    )
    audio_bytes = audio_recording["bytes"] if audio_recording and audio_recording.get("bytes") else None

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Send voice note", key="send_voice"):
            with st.spinner("Processing voice note..."):
                result = app.send_voice_message(st.session_state.conversation_id, audio_bytes)

                if result.get("success"):
                    user_transcript = result.get(
                        "user_transcript",
                        "Transcript not available for this audio.",
                    )
                    st.session_state.messages.append(
                        {
                            "type": "user",
                            "content": f"(Voice) {user_transcript}",
                            "timestamp": datetime.now(),
                        }
                    )

                    ai_response = result.get(
                        "ai_response_text",
                        "No response received. Please try sending again.",
                    )
                    st.session_state.messages.append(
                        {
                            "type": "ai",
                            "content": ai_response,
                            "timestamp": datetime.now(),
                        }
                    )

                    st.success("Voice note sent.")
                    st.rerun()
                else:
                    st.error(
                        f"Unable to send voice note: {result.get('error', 'Unknown error')}"
                    )

    st.markdown("---")
    col_control1, col_control2 = st.columns(2)

    with col_control1:
        if st.button("Start over", key="new_conversation"):
            app.end_conversation(st.session_state.conversation_id)
            st.session_state.conversation_id = None
            st.session_state.conversation_started = False
            st.session_state.messages = []
            st.session_state.current_stage = "welcome"
            st.rerun()

    with col_control2:
        if st.button("Finish conversation", key="end_conversation"):
            with st.spinner("Closing conversation..."):
                result = app.end_conversation(st.session_state.conversation_id)

                if result.get("success"):
                    st.session_state.current_stage = "completed"
                    st.rerun()
                else:
                    st.error("Could not end the conversation. Please try again.")

# Completion stage
elif st.session_state.current_stage == "completed":
    st.success(
        "Thank you for sharing your feedback. Your conversation has been submitted successfully."
    )

    st.markdown(
        """
    <div class="instruction">
        <h3>What happens next</h3>
        <p>Our team will review your feedback and follow up if contact information is available.</p>
        <p><strong>Tips for a clear report:</strong></p>
        <ul>
            <li>Keep the description short and specific.</li>
            <li>Mention the location or nearby landmark.</li>
            <li>Include any urgent details like safety risks.</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.button("Start a new feedback session"):
        for key in [
            "conversation_id",
            "conversation_started",
            "messages",
            "user_name",
            "village_name",
            "current_stage",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>JanSpandana.AI v1.0 | Voice feedback helper | Support: support@janspandana.gov.in</p>
</div>
""",
    unsafe_allow_html=True,
)
